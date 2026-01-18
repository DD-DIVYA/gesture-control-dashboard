import React from 'react';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Square } from 'lucide-react';

interface WheelchairControlsProps {
  mode: 'STOP' | 'WHEELCHAIR' | 'PLACE';
  lastDirection: string;
  motorSpeed: number;
  movementIntensity: number;
}

interface DirectionButton {
  direction: string;
  icon: React.ReactNode;
  label: string;
  position: string;
}

export const WheelchairControls: React.FC<WheelchairControlsProps> = ({
  mode,
  lastDirection,
  motorSpeed,
  movementIntensity,
}) => {
  const isActive = mode === 'WHEELCHAIR';
  
  const directions: DirectionButton[] = [
    { direction: 'FORWARD', icon: <ArrowUp className="w-8 h-8" />, label: 'Forward', position: 'col-span-1 col-start-2' },
    { direction: 'LEFT', icon: <ArrowLeft className="w-8 h-8" />, label: 'Left', position: 'col-span-1 row-start-2' },
    { direction: 'STOP', icon: <Square className="w-8 h-8" />, label: 'Stop', position: 'col-span-1 col-start-2 row-start-2' },
    { direction: 'RIGHT', icon: <ArrowRight className="w-8 h-8" />, label: 'Right', position: 'col-span-1 col-start-3 row-start-2' },
    { direction: 'BACKWARD', icon: <ArrowDown className="w-8 h-8" />, label: 'Backward', position: 'col-span-1 col-start-2 row-start-3' },
  ];

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 transition-all duration-300 ${
      !isActive ? 'opacity-40 pointer-events-none' : ''
    }`}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Wheelchair Controls</h2>
        <p className="text-gray-600">Head-controlled movement directions</p>
        {isActive && (
          <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <div className="text-sm text-blue-800 font-semibold mb-1">🎯 Active Controls</div>
            <div className="text-xs text-blue-600">
              • Move your head to control wheelchair direction<br/>
              • <strong>Long blink</strong> (hold) → Return to STOP mode
            </div>
          </div>
        )}
        {!isActive && (
          <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-600 font-semibold mb-1">⚡ Activation</div>
            <div className="text-xs text-gray-500">
              <strong>Single blink</strong> → Activate wheelchair controls
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 grid-rows-3 gap-4 max-w-md mx-auto">
        {directions.map((dir) => {
          const isCurrentDirection = lastDirection.toUpperCase() === dir.direction;
          const isStopActive = lastDirection.toUpperCase() === 'STOP' && dir.direction === 'STOP';
          const shouldHighlight = isCurrentDirection || isStopActive;
          
          return (
            <button
              key={dir.direction}
              className={`
                ${dir.position}
                flex flex-col items-center justify-center
                p-6 rounded-2xl border-2 transition-all duration-200
                min-h-[120px] min-w-[120px]
                ${shouldHighlight
                  ? 'border-green-500 bg-green-50 text-green-700 shadow-lg shadow-green-500/25 scale-105'
                  : 'border-gray-200 bg-gray-50 text-gray-600 hover:border-gray-300 hover:bg-gray-100'
                }
              `}
              disabled={!isActive}
            >
              <div className="mb-2">
                {dir.icon}
              </div>
              <span className="text-sm font-semibold">{dir.label}</span>
            </button>
          );
        })}
      </div>

      <div className="mt-6 space-y-4">
        <div className="text-center">
          <div className="text-sm text-gray-500">
            Current Direction: <span className="font-semibold text-gray-700 capitalize">{lastDirection}</span>
          </div>
        </div>
        
        {/* Real-time movement metrics */}
        {isActive && (
          <div className="bg-gray-50 rounded-lg p-4 space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">Motor Speed</span>
              <span className="text-sm font-bold text-blue-600">{Math.round(motorSpeed)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(0, Math.min(100, motorSpeed))}%` }}
              ></div>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">Movement Intensity</span>
              <span className="text-sm font-bold text-purple-600">{Math.round(movementIntensity * 100)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-purple-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.max(0, Math.min(100, movementIntensity * 100))}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};