import React from 'react';
import { Eye, Square, Navigation, MapPin, Info } from 'lucide-react';

interface BlinkControlStatusProps {
  mode: 'STOP' | 'WHEELCHAIR' | 'PLACE';
  lastDirection: string;
  highlightedPlace: string | null;
  selectedPlace: string | null;
}

export const BlinkControlStatus: React.FC<BlinkControlStatusProps> = ({
  mode,
  lastDirection,
  highlightedPlace,
  selectedPlace,
}) => {
  const getModeInfo = () => {
    switch (mode) {
      case 'STOP':
        return {
          icon: <Square className="w-6 h-6" />,
          title: 'STOP Mode',
          status: 'Ready for activation',
          color: 'bg-gray-100 text-gray-700',
          borderColor: 'border-gray-300',
          instructions: [
            'Single blink → Activate wheelchair controls',
            'Double blink → Activate place selection'
          ]
        };
      case 'WHEELCHAIR':
        return {
          icon: <Navigation className="w-6 h-6" />,
          title: 'Wheelchair Mode',
          status: `Moving ${lastDirection.toLowerCase()}`,
          color: 'bg-blue-100 text-blue-700',
          borderColor: 'border-blue-300',
          instructions: [
            'Move head → Control wheelchair direction',
            'Long blink (hold) → Return to STOP mode'
          ]
        };
      case 'PLACE':
        return {
          icon: <MapPin className="w-6 h-6" />,
          title: 'Place Selection Mode',
          status: highlightedPlace ? `Highlighting: ${highlightedPlace}` : 'Navigating places',
          color: 'bg-purple-100 text-purple-700',
          borderColor: 'border-purple-300',
          instructions: [
            'Single blink → Navigate through places',
            'Double blink (within 1.2s) → Select highlighted place',
            'Long blink (hold 0.7s+) → Return to STOP mode'
          ]
        };
      default:
        return {
          icon: <Eye className="w-6 h-6" />,
          title: 'Unknown Mode',
          status: 'Status unknown',
          color: 'bg-gray-100 text-gray-700',
          borderColor: 'border-gray-300',
          instructions: []
        };
    }
  };

  const modeInfo = getModeInfo();

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 border-2 ${modeInfo.borderColor} transition-all duration-300`}>
      <div className="flex items-center gap-4 mb-4">
        <div className={`p-3 rounded-lg ${modeInfo.color}`}>
          {modeInfo.icon}
        </div>
        <div>
          <h2 className="text-xl font-bold text-gray-900">{modeInfo.title}</h2>
          <p className="text-gray-600">{modeInfo.status}</p>
        </div>
      </div>

      {/* Current Selection Info */}
      {selectedPlace && (
        <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2 text-green-800">
            <MapPin className="w-4 h-4" />
            <span className="font-semibold">Selected: {selectedPlace}</span>
          </div>
          <p className="text-xs text-green-600 mt-1">Ready for navigation</p>
        </div>
      )}

      {/* Blink Instructions */}
      <div className="bg-gray-50 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Info className="w-4 h-4 text-gray-600" />
          <span className="text-sm font-semibold text-gray-700">Blink Controls</span>
        </div>
        <div className="space-y-2">
          {modeInfo.instructions.map((instruction, index) => (
            <div key={index} className="flex items-start gap-2">
              <div className="w-1 h-1 bg-gray-400 rounded-full mt-2 flex-shrink-0"></div>
              <span className="text-xs text-gray-600">{instruction}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Blink Detection Visual Feedback */}
      <div className="mt-4 space-y-3">
        <div className="flex items-center justify-center">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Eye className="w-4 h-4" />
            <span>Blink detection active</span>
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          </div>
        </div>
        
        {/* Blink Type Legend */}
        <div className="bg-white border border-gray-200 rounded-lg p-3">
          <div className="text-xs font-semibold text-gray-700 mb-2">Blink Types:</div>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="flex flex-col items-center p-2 bg-blue-50 rounded">
              <div className="w-8 h-3 bg-blue-300 rounded mb-1 animate-pulse"></div>
              <span className="text-blue-700 font-medium">Single</span>
              <span className="text-blue-600">Quick</span>
            </div>
            <div className="flex flex-col items-center p-2 bg-purple-50 rounded">
              <div className="flex gap-1 mb-1">
                <div className="w-3 h-3 bg-purple-300 rounded animate-pulse"></div>
                <div className="w-3 h-3 bg-purple-300 rounded animate-pulse" style={{animationDelay: '0.3s'}}></div>
              </div>
              <span className="text-purple-700 font-medium">Double</span>
              <span className="text-purple-600">Within 1.2s</span>
            </div>
            <div className="flex flex-col items-center p-2 bg-red-50 rounded">
              <div className="w-8 h-3 bg-red-300 rounded mb-1"></div>
              <span className="text-red-700 font-medium">Long</span>
              <span className="text-red-600">Hold 0.7s+</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};