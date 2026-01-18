import { useEffect, useRef, useState } from 'react';

interface CameraStreamProps {
  onFaceDetection?: (result: any) => void;
  sendMessage?: (message: any) => void;
  connected?: boolean;
}

const CameraStream = ({ onFaceDetection, sendMessage, connected }: CameraStreamProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, []);

  useEffect(() => {
    if (connected && isStreaming && sendMessage) {
      startFrameCapture();
    } else {
      stopFrameCapture();
    }
  }, [connected, isStreaming, sendMessage]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { 
          width: 640, 
          height: 480,
          frameRate: 15
        }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play();
          setIsStreaming(true);
        };
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
    }
    setIsStreaming(false);
    stopFrameCapture();
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current || !sendMessage) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (!ctx) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw current frame to canvas
    ctx.drawImage(video, 0, 0);

    // Convert to base64
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    // Send frame to backend for processing
    sendMessage({
      type: 'camera_frame',
      image: imageData,
      timestamp: Date.now()
    });
  };

  const startFrameCapture = () => {
    stopFrameCapture(); // Clear any existing interval
    // Capture frames every 200ms (5 FPS to avoid overload)
    intervalRef.current = setInterval(captureFrame, 200);
  };

  const stopFrameCapture = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = undefined;
    }
  };

  // Handle face detection results from backend
  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'face_detection_result') {
          setFaceDetected(data.payload.faces_detected);
          onFaceDetection?.(data.payload);
        }
      } catch (e) {
        // Ignore non-JSON messages
      }
    };

    // This is a simplified approach - in real implementation, 
    // you'd get this through the WebSocket hook
    if (typeof window !== 'undefined') {
      window.addEventListener('message', handleMessage);
      return () => window.removeEventListener('message', handleMessage);
    }
  }, [onFaceDetection]);

  return (
    <div className="camera-stream relative">
      <div className="relative rounded-lg overflow-hidden bg-gray-900">
        <video
          ref={videoRef}
          className="w-full h-48 object-cover"
          muted
          playsInline
        />
        
        {/* Face detection indicator */}
        <div className={`absolute top-2 left-2 px-2 py-1 rounded text-xs font-bold ${
          faceDetected 
            ? 'bg-green-500 text-white' 
            : 'bg-red-500 text-white'
        }`}>
          {faceDetected ? '👁️ Face Detected' : '🔍 Looking for face...'}
        </div>

        {/* Connection status */}
        <div className={`absolute top-2 right-2 px-2 py-1 rounded text-xs ${
          connected && isStreaming
            ? 'bg-blue-500 text-white'
            : 'bg-gray-500 text-white'
        }`}>
          {connected && isStreaming ? '🔴 LIVE' : '⭕ Offline'}
        </div>

        {/* Hidden canvas for frame capture */}
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
      </div>

      <div className="mt-2 text-center">
        <p className="text-sm text-gray-600">
          {isStreaming 
            ? `📷 Camera active • ${connected ? 'Processing frames' : 'Waiting for connection'}`
            : '📷 Camera starting...'
          }
        </p>
      </div>
    </div>
  );
};

export default CameraStream;