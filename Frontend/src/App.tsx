import React, { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { StatusBar } from './components/StatusBar';
import { WheelchairControls } from './components/WheelchairControls';
import { PlacesPanel } from './components/PlacesPanel';
import { NotificationToast } from './components/NotificationToast';
import { MetricsDashboard } from './components/MetricsDashboard';
import { BlinkControlStatus } from './components/BlinkControlStatus';
import CameraStream from './components/CameraStream';
import { Accessibility, Camera, CameraOff } from 'lucide-react';

// WebSocket URL - connect to cloud backend for real camera detection
const WS_URL = import.meta.env.VITE_WS_URL || 'wss://gesture-control-dashboard.onrender.com/ws';

function App() {
  const { state, lastHeadDirection, notifications, sendMessage, connect, removeNotification } = useWebSocket(WS_URL);
  const [showCamera, setShowCamera] = useState(true);
  const [faceDetectionData, setFaceDetectionData] = useState<any>(null);

  const handleCalibrateHead = () => {
    sendMessage({ event: 'CALIBRATE' });
  };

  const handleCalibrateEyes = () => {
    sendMessage({ event: 'CALIBRATE_EYES' });
  };

  const handleResetPlaces = () => {
    sendMessage({ event: 'RESET_PLACES' });
  };

  const handleFaceDetection = (result: any) => {
    setFaceDetectionData(result);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      {/* Status Bar */}
      <StatusBar
        connected={state.connected}
        mode={state.mode}
        onCalibrateHead={handleCalibrateHead}
        onCalibrateEyes={handleCalibrateEyes}
        onResetPlaces={handleResetPlaces}
      />

      {/* Main Content */}
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-blue-100 rounded-full">
              <Accessibility className="w-8 h-8 text-blue-600" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900">
              Wheelchair Controller
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Navigate and control your wheelchair using head movements and eye tracking for seamless mobility assistance.
          </p>
        </div>

        {/* Connection Status Message */}
        {!state.connected && (
          <div className="mb-8 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center justify-between">
              <p className="text-yellow-800">
                {state.connecting ? (
                  <>
                    <span className="font-semibold">Connecting...</span> Please wait
                  </>
                ) : (
                  <>
                    <span className="font-semibold">Disconnected.</span> Make sure your wheelchair controller is running on {WS_URL}
                  </>
                )}
              </p>
              {!state.connecting && (
                <button
                  onClick={connect}
                  className="ml-4 px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 transition-colors text-sm font-medium"
                >
                  Reconnect
                </button>
              )}
            </div>
          </div>
        )}

        {/* Mode Status Display */}
        {state.connected && (
          <div className="mb-8 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-center">
              <h3 className="text-lg font-semibold text-blue-900 mb-2">
                Current Mode: <span className="uppercase">{state.mode}</span>
              </h3>
              <div className="text-sm text-blue-700">
                {state.mode === 'STOP' && (
                  <p>System is in standby. Use blinks to activate: <strong>1 blink</strong> for wheelchair control, <strong>2 blinks</strong> for place selection.</p>
                )}
                {state.mode === 'WHEELCHAIR' && (
                  <p>Head movement controls active. Use <strong>long blink</strong> to return to STOP.</p>
                )}
                {state.mode === 'PLACE' && (
                  <p>Place selection mode. <strong>1 blink</strong> to navigate, <strong>2 blinks</strong> to select, <strong>long blink</strong> to stop.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* System Metrics Dashboard */}
        {state.connected && (
          <div className="mb-8 max-w-7xl mx-auto">
            <MetricsDashboard
              batteryPercentage={state.batteryPercentage}
              motorSpeed={state.motorSpeed}
              movementIntensity={state.movementIntensity}
              totalDistance={state.totalDistance}
              sessionTime={state.sessionTime}
              faceTracking={state.faceTracking}
            />
          </div>
        )}

        {/* Blink Control Status */}
        {state.connected && (
          <div className="mb-8 max-w-md mx-auto">
            <BlinkControlStatus
              mode={state.mode}
              lastDirection={lastHeadDirection}
              highlightedPlace={state.highlight}
              selectedPlace={state.selected}
            />
          </div>
        )}

        {/* Camera and Face Detection Section */}
        <div className="mb-8 max-w-4xl mx-auto">
          <div className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center gap-2">
                <Camera className="w-5 h-5" />
                Live Camera Feed
              </h2>
              <button
                onClick={() => setShowCamera(!showCamera)}
                className={`flex items-center gap-2 px-3 py-1 rounded text-sm font-medium ${
                  showCamera 
                    ? 'bg-red-100 text-red-700 hover:bg-red-200'
                    : 'bg-green-100 text-green-700 hover:bg-green-200'
                }`}
              >
                {showCamera ? <CameraOff className="w-4 h-4" /> : <Camera className="w-4 h-4" />}
                {showCamera ? 'Hide Camera' : 'Show Camera'}
              </button>
            </div>
            
            {showCamera && (
              <div className="grid md:grid-cols-2 gap-6">
                <CameraStream
                  onFaceDetection={handleFaceDetection}
                  sendMessage={sendMessage}
                  connected={state.connected}
                />
                
                {/* Face Detection Info */}
                <div className="space-y-4">
                  <h3 className="font-medium text-gray-900">Face Detection Status</h3>
                  
                  <div className={`p-3 rounded-lg ${
                    faceDetectionData?.faces_detected 
                      ? 'bg-green-50 border border-green-200'
                      : 'bg-red-50 border border-red-200'
                  }`}>
                    <div className="flex items-center gap-2 mb-2">
                      <div className={`w-3 h-3 rounded-full ${
                        faceDetectionData?.faces_detected ? 'bg-green-500' : 'bg-red-500'
                      }`} />
                      <span className="font-medium">
                        {faceDetectionData?.faces_detected ? 'Face Detected!' : 'Looking for face...'}
                      </span>
                    </div>
                    
                    {faceDetectionData && (
                      <div className="text-sm text-gray-600 space-y-1">
                        <div>Faces found: {faceDetectionData.face_count || 0}</div>
                        <div>Status: {faceDetectionData.status}</div>
                        {faceDetectionData.faces && faceDetectionData.faces.length > 0 && (
                          <div>
                            Confidence: {(faceDetectionData.faces[0].confidence * 100).toFixed(1)}%
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="text-xs text-gray-500">
                    <p>• Face detection is processed in real-time</p>
                    <p>• Your video data is processed securely in the cloud</p>
                    <p>• No video data is stored or recorded</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Main Panels */}
        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Wheelchair Controls */}
          <div className="space-y-6">
            <WheelchairControls 
              mode={state.mode} 
              lastDirection={lastHeadDirection}
              motorSpeed={state.motorSpeed}
              movementIntensity={state.movementIntensity}
            />
          </div>

          {/* Places Panel */}
          <div className="space-y-6">
            <PlacesPanel
              mode={state.mode}
              rooms={state.rooms}
              highlight={state.highlight}
              selected={state.selected}
            />
          </div>
        </div>

        {/* Footer Info */}
        <div className="mt-12 text-center text-sm text-gray-500">
          <p>
            Use blinks to control modes • Head movements for wheelchair navigation • Long blink to stop/reset
          </p>
        </div>
      </div>

      {/* Notifications */}
      <NotificationToast notifications={notifications} onRemove={removeNotification} />
    </div>
  );
}

export default App;