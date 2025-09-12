import React from 'react';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, Square } from 'lucide-react';

interface WheelchairControlsProps {
  mode: 'wheelchair' | 'place';
  lastDirection: string;
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
}) => {
  const isActive = mode === 'wheelchair';
  
  const directions: DirectionButton[] = [
    { direction: 'forward', icon: <ArrowUp className="w-8 h-8" />, label: 'Forward', position: 'col-span-1 col-start-2' },
    { direction: 'left', icon: <ArrowLeft className="w-8 h-8" />, label: 'Left', position: 'col-span-1 row-start-2' },
    { direction: 'stop', icon: <Square className="w-8 h-8" />, label: 'Stop', position: 'col-span-1 col-start-2 row-start-2' },
    { direction: 'right', icon: <ArrowRight className="w-8 h-8" />, label: 'Right', position: 'col-span-1 col-start-3 row-start-2' },
    { direction: 'backward', icon: <ArrowDown className="w-8 h-8" />, label: 'Backward', position: 'col-span-1 col-start-2 row-start-3' },
  ];

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 transition-all duration-300 ${
      !isActive ? 'opacity-40 pointer-events-none' : ''
    }`}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Wheelchair Controls</h2>
        <p className="text-gray-600">Head-controlled movement directions</p>
      </div>

      <div className="grid grid-cols-3 grid-rows-3 gap-4 max-w-md mx-auto">
        {directions.map((dir) => {
          const isCurrentDirection = lastDirection === dir.direction;
          const isStopActive = lastDirection === 'stop' && dir.direction === 'stop';
          const shouldHighlight = isCurrentDirection || (isStopActive && lastDirection !== 'stop');
          
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

      <div className="mt-6 text-center">
        <div className="text-sm text-gray-500">
          Current Direction: <span className="font-semibold text-gray-700 capitalize">{lastDirection}</span>
        </div>
      </div>
    </div>
  );
};