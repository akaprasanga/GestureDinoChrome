importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs');
importScripts('https://cdn.jsdelivr.net/npm/@tensorflow-models/handpose');

let handposeModel = null;

async function initModel() {
    try {
        handposeModel = await handpose.load();
        postMessage({ type: 'modelLoaded' });
    } catch (error) {
        postMessage({ type: 'error', error: error.message });
    }
}

async function detectHands(imageData, width, height) {
    if (!handposeModel) {
        postMessage({ type: 'error', error: 'Model not loaded' });
        return;
    }
    
    try {
        const input = tf.tidy(() => {
            const img = tf.browser.fromPixels(
                { data: new Uint8Array(imageData.data), width: width, height: height },
                4
            );
            return tf.slice3d(img, [0, 0, 0], [-1, -1, 3]);
        });

        const predictions = await handposeModel.estimateHands(input);
        input.dispose();
        
        if (predictions.length > 0) {
            const keypoints = predictions[0].annotations;
            
            // Get fingertip and middle knuckle positions for each finger
            const fingerStates = {
                thumb: [keypoints.thumb[3], keypoints.thumb[1]],
                index: [keypoints.indexFinger[3], keypoints.indexFinger[1]],
                middle: [keypoints.middleFinger[3], keypoints.middleFinger[1]],
                ring: [keypoints.ringFinger[3], keypoints.ringFinger[1]],
                pinky: [keypoints.pinky[3], keypoints.pinky[1]]
            };

            // Calculate if each finger is extended by comparing tip to knuckle position
            let extendedFingers = 0;
            for (const [finger, [tip, knuckle]] of Object.entries(fingerStates)) {
                // For thumb, check horizontal distance (x-axis) from knuckle
                if (finger === 'thumb') {
                    const distance = Math.abs(tip[0] - knuckle[0]);
                    if (distance > 30) extendedFingers++;
                } 
                // For other fingers, check vertical distance (y-axis) from knuckle
                else {
                    if (tip[1] < knuckle[1] - 30) extendedFingers++;
                }
            }

            let gesture = "Unknown";
            if (extendedFingers >= 4) {
                gesture = "Open Hand";
            } else if (extendedFingers <= 1) {
                gesture = "Closed Hand";
            }
            
            console.log('Detected gesture:', gesture, 'Extended fingers:', extendedFingers);
            postMessage({ type: 'gestureDetected', gesture });
        } else {
            postMessage({ type: 'gestureDetected', gesture: 'No hand detected' });
        }
    } catch (error) {
        console.error('Detection error:', error);
        postMessage({ type: 'error', error: error.message });
    }
    
    postMessage({ type: 'processingComplete' });
}

self.onmessage = async function(e) {
    switch (e.data.type) {
        case 'init':
            await initModel();
            break;
        case 'detect':
            await detectHands(e.data.imageData, e.data.width, e.data.height);
            break;
    }
}; 