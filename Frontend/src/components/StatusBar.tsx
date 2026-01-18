import React from 'react';
import { Wifi, WifiOff, RotateCcw, Eye, RotateCw } from 'lucide-react';

interface StatusBarProps {
  connected: boolean;
  mode: 'STOP' | 'WHEELCHAIR' | 'PLACE';
  onCalibrateHead: () => void;
  onCalibrateEyes: () => void;
  onResetPlaces: () => void;
}

export const StatusBar: React.FC<StatusBarProps> = ({
  connected,
  mode,
  onCalibrateHead,
  onCalibrateEyes,
  onResetPlaces,
}) => {
  return (
    <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-6">
          {/* Connection Status */}
          <div className="flex items-center gap-2">
            <div className="relative">
              {connected ? (
                <>
                  <Wifi className="w-5 h-5 text-green-500" />
                  <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                </>
              ) : (
                <WifiOff className="w-5 h-5 text-red-500" />
              )}
            </div>
            <span className={`text-sm font-medium ${
              connected ? 'text-green-700' : 'text-red-700'
            }`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Mode Badge */}
          <div className={`px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${
            mode === 'WHEELCHAIR'
              ? 'bg-blue-100 text-blue-800'
              : mode === 'PLACE'
              ? 'bg-purple-100 text-purple-800'
              : 'bg-gray-100 text-gray-800'
          }`}>
            {mode}
          </div>
        </div>

        {/* Control Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={onCalibrateHead}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Calibrate Head"
          >
            <RotateCcw className="w-4 h-4" />
            <span className="hidden sm:inline">Calibrate Head</span>
          </button>
          
          <button
            onClick={onCalibrateEyes}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Calibrate Eyes"
          >
            <Eye className="w-4 h-4" />
            <span className="hidden sm:inline">Calibrate Eyes</span>
          </button>
          
          <button
            onClick={onResetPlaces}
            className="flex items-center gap-2 px-3 py-2 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
            title="Reset Places"
          >
            <RotateCw className="w-4 h-4" />
            <span className="hidden sm:inline">Reset Places</span>
          </button>
        </div>
      </div>
    </div>
  );
};