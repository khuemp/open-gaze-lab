/**
 * Reads fps / resolution / frame count from a scene video, in the browser.
 *
 * This replaces the one place the pipeline used OpenCV (`cv2.VideoCapture` in
 * preprocess_headmounted/common.py). OpenCV's wasm build ships no video codecs
 * and cannot demux MP4 at all, and OpenCV.js has the same limitation — it
 * relies on a <video> element for frames. So the container is parsed directly
 * with mp4box.js, which reads the `moov` atom and needs no decoding.
 *
 * fps is derived as `nb_samples / (duration / timescale)`. For constant-frame-
 * rate files that is exact; for variable-frame-rate files it is the average
 * rate — which is also what OpenCV reports, since CAP_PROP_FPS maps to
 * FFmpeg's avg_frame_rate. So the numbers match the previous behaviour.
 */

import { createFile } from 'mp4box';

const CHUNK_SIZE = 1 << 20;        // 1 MiB
const HEAD_SCAN_LIMIT = 32 << 20;  // give up scanning forward after 32 MiB
const TAIL_SCAN_LIMIT = 64 << 20;  // then try the tail, for non-faststart files
const ELEMENT_TIMEOUT_MS = 15000;

/**
 * Probe *file* for the metadata the head-mounted pipeline needs.
 *
 * Never rejects on an unreadable container: callers get whatever could be
 * determined, with `fps: 0` meaning "ask the user". GiW datasets hard-fail
 * without a real fps, so the form exposes an editable field.
 *
 * @returns {Promise<{fps:number,width:number,height:number,duration_s:number,
 *   n_frames:number,source:string,warning?:string}>}
 */
export async function probeVideoMetadata(file) {
    try {
        const parsed = await probeWithMp4Box(file);
        if (parsed) return parsed;
    } catch (error) {
        console.warn('mp4box probe failed, falling back to <video>', error);
    }

    try {
        return await probeWithVideoElement(file);
    } catch (error) {
        console.warn('<video> probe failed', error);
        return {
            fps: 0,
            width: 0,
            height: 0,
            duration_s: 0,
            n_frames: 0,
            source: 'none',
            warning: 'Could not read the video metadata. Enter the frame rate and resolution manually.',
        };
    }
}

/** Parse the moov atom without decoding a single frame. */
function probeWithMp4Box(file) {
    return new Promise((resolve, reject) => {
        const mp4 = createFile(false);
        let settled = false;

        const finish = (value) => {
            if (settled) return;
            settled = true;
            resolve(value);
        };

        mp4.onError = (error) => {
            if (!settled) {
                settled = true;
                reject(new Error(typeof error === 'string' ? error : 'MP4 parse error'));
            }
        };

        mp4.onReady = (info) => {
            const track = info.tracks.find((t) => t.video);
            if (!track) {
                finish(null); // audio-only or unrecognised: let the fallback try
                return;
            }

            const durationSeconds = track.timescale > 0 ? track.duration / track.timescale : 0;
            const fps = durationSeconds > 0 ? track.nb_samples / durationSeconds : 0;

            finish({
                fps,
                width: track.video.width,
                height: track.video.height,
                duration_s: durationSeconds,
                n_frames: track.nb_samples,
                source: 'mp4box',
            });
        };

        feedChunks(file, mp4, () => settled).then(() => finish(null)).catch(reject);
    });
}

/**
 * Stream *file* into the parser until it has enough to report.
 *
 * Scans forward first, since well-formed streaming MP4s put `moov` at the
 * front. Files written without faststart keep it at the very end, so the tail
 * is tried next rather than reading a multi-gigabyte recording end to end.
 */
async function feedChunks(file, mp4, isDone) {
    const headEnd = Math.min(file.size, HEAD_SCAN_LIMIT);
    for (let offset = 0; offset < headEnd; offset += CHUNK_SIZE) {
        if (isDone()) return;
        await appendSlice(file, mp4, offset, Math.min(offset + CHUNK_SIZE, headEnd));
        if (isDone()) return;
    }

    const tailStart = Math.max(headEnd, file.size - TAIL_SCAN_LIMIT);
    for (let offset = tailStart; offset < file.size; offset += CHUNK_SIZE) {
        if (isDone()) return;
        await appendSlice(file, mp4, offset, Math.min(offset + CHUNK_SIZE, file.size));
        if (isDone()) return;
    }

    mp4.flush();
}

async function appendSlice(file, mp4, start, end) {
    const buffer = await file.slice(start, end).arrayBuffer();
    // mp4box identifies each buffer's position in the file by this property.
    buffer.fileStart = start;
    mp4.appendBuffer(buffer);
}

/**
 * Fallback: let the browser's own demuxer report what it can.
 *
 * Gives exact dimensions and duration but no frame count, so fps has to be
 * sampled from actual playback via requestVideoFrameCallback. That needs the
 * video to play, which autoplay policies may refuse — hence `fps: 0` being an
 * expected outcome rather than an error.
 */
function probeWithVideoElement(file) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        const url = URL.createObjectURL(file);
        video.preload = 'metadata';
        video.muted = true;
        video.playsInline = true;

        const cleanup = () => {
            video.removeAttribute('src');
            video.load();
            URL.revokeObjectURL(url);
        };

        const timer = setTimeout(() => {
            cleanup();
            reject(new Error('Timed out reading video metadata'));
        }, ELEMENT_TIMEOUT_MS);

        video.addEventListener('error', () => {
            clearTimeout(timer);
            cleanup();
            reject(new Error('The browser could not read this video file'));
        });

        video.addEventListener('loadedmetadata', async () => {
            clearTimeout(timer);
            const duration = Number.isFinite(video.duration) ? video.duration : 0;
            const fps = await sampleFrameRate(video);
            const result = {
                fps,
                width: video.videoWidth,
                height: video.videoHeight,
                duration_s: duration,
                n_frames: fps > 0 ? Math.round(fps * duration) : 0,
                source: 'video-element',
            };
            if (fps === 0) {
                result.warning =
                    'Could not determine the frame rate automatically. Enter it manually.';
            }
            cleanup();
            resolve(result);
        });

        video.src = url;
    });
}

/** Estimate fps from two requestVideoFrameCallback samples. Returns 0 on failure. */
function sampleFrameRate(video) {
    if (typeof video.requestVideoFrameCallback !== 'function') return Promise.resolve(0);

    return new Promise((resolve) => {
        let first = null;
        const done = (value) => {
            video.pause();
            resolve(value);
        };
        const timer = setTimeout(() => done(0), 3000);

        const onFrame = (_now, metadata) => {
            if (!first) {
                first = metadata;
                video.requestVideoFrameCallback(onFrame);
                return;
            }
            const elapsed = metadata.mediaTime - first.mediaTime;
            const frames = metadata.presentedFrames - first.presentedFrames;
            // Require a real interval; back-to-back callbacks give a useless ratio.
            if (elapsed > 0.25 && frames > 0) {
                clearTimeout(timer);
                done(frames / elapsed);
            } else {
                video.requestVideoFrameCallback(onFrame);
            }
        };

        video.requestVideoFrameCallback(onFrame);
        video.play().catch(() => {
            clearTimeout(timer);
            resolve(0); // autoplay blocked — the user will type the fps
        });
    });
}
