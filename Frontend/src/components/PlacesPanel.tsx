import React from 'react';
import { MapPin, CheckCircle2, Navigation } from 'lucide-react';

interface PlacesPanelProps {
  mode: 'STOP' | 'WHEELCHAIR' | 'PLACE';
  rooms: string[];
  highlight: string | null;
  selected: string | null;
}

export const PlacesPanel: React.FC<PlacesPanelProps> = ({
  mode,
  rooms,
  highlight,
  selected,
}) => {
  const isActive = mode === 'PLACE';

  return (
    <div className={`bg-white rounded-xl shadow-lg p-6 transition-all duration-300 ${
      !isActive ? 'opacity-40 pointer-events-none' : ''
    }`}>
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Places</h2>
        <p className="text-gray-600">Eye-controlled destination selection</p>
        {isActive && (
          <div className="mt-3 p-3 bg-purple-50 border border-purple-200 rounded-lg">
            <div className="text-sm text-purple-800 font-semibold mb-1">🎯 Active Controls</div>
            <div className="text-xs text-purple-600">
              • <strong>Single blink</strong> → Navigate through places<br/>
              • <strong>Double blink</strong> → Select highlighted place<br/>
              • <strong>Long blink</strong> (hold) → Return to STOP mode
            </div>
          </div>
        )}
        {!isActive && (
          <div className="mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="text-sm text-gray-600 font-semibold mb-1">⚡ Activation</div>
            <div className="text-xs text-gray-500">
              <strong>Double blink</strong> → Activate place selection
            </div>
          </div>
        )}
      </div>

      {rooms.length === 0 ? (
        <div className="text-center py-12">
          <MapPin className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <p className="text-gray-500">No places available</p>
        </div>
      ) : (
        <div className="space-y-3">
          {rooms.map((room) => {
            const isHighlighted = highlight === room;
            const isSelected = selected === room;
            
            return (
              <div
                key={room}
                className={`
                  flex items-center justify-between
                  p-4 rounded-xl border-2 transition-all duration-200
                  ${isHighlighted
                    ? 'border-yellow-400 bg-yellow-50 shadow-lg shadow-yellow-400/25'
                    : isSelected
                    ? 'border-green-500 bg-green-50 shadow-lg shadow-green-500/25'
                    : 'border-gray-200 bg-gray-50'
                  }
                `}
              >
                <div className="flex items-center gap-3">
                  <div className={`
                    p-2 rounded-lg
                    ${isHighlighted
                      ? 'bg-yellow-100 text-yellow-700'
                      : isSelected
                      ? 'bg-green-100 text-green-700'
                      : 'bg-gray-100 text-gray-600'
                    }
                  `}>
                    <MapPin className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className={`font-semibold ${
                      isHighlighted ? 'text-yellow-900' : isSelected ? 'text-green-900' : 'text-gray-900'
                    }`}>
                      {room}
                    </h3>
                    {isHighlighted && (
                      <p className="text-sm text-yellow-600">Highlighted</p>
                    )}
                    {isSelected && (
                      <p className="text-sm text-green-600">Selected</p>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  {isSelected && (
                    <div className="flex items-center gap-1 px-2 py-1 bg-green-100 text-green-700 rounded-full text-xs font-semibold">
                      <CheckCircle2 className="w-3 h-3" />
                      Ready
                    </div>
                  )}
                  {isHighlighted && !isSelected && (
                    <div className="flex items-center gap-1 px-2 py-1 bg-yellow-100 text-yellow-700 rounded-full text-xs font-semibold">
                      <Navigation className="w-3 h-3" />
                      Active
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {selected && (
        <div className="mt-6 p-4 bg-green-50 border border-green-200 rounded-lg">
          <div className="flex items-center gap-2 text-green-800">
            <CheckCircle2 className="w-5 h-5" />
            <span className="font-semibold">Ready to go to {selected}</span>
          </div>
          <p className="text-sm text-green-600 mt-1">
            Blink twice more to navigate to this location
          </p>
        </div>
      )}
    </div>
  );
};