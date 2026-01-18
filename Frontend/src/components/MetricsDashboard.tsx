import React from 'react';
import { Battery, Zap, Activity, Clock, MapPin, Eye } from 'lucide-react';

interface MetricsDashboardProps {
  batteryPercentage: number;
  motorSpeed: number;
  movementIntensity: number;
  totalDistance: number;
  sessionTime: number;
  faceTracking: boolean;
}

const MetricCard: React.FC<{
  icon: React.ReactNode;
  label: string;
  value: string | number;
  unit?: string;
  color?: string;
  statusIndicator?: boolean;
}> = ({ icon, label, value, unit, color = 'blue', statusIndicator = false }) => {
  return (
    <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-4">
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg bg-${color}-100`}>
          <div className={`w-5 h-5 text-${color}-600`}>{icon}</div>
        </div>
        {statusIndicator && (
          <div className={`w-2 h-2 rounded-full ${
            typeof value === 'boolean' ? (value ? 'bg-green-400' : 'bg-red-400') : 'bg-gray-300'
          }`}></div>
        )}
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium text-gray-600">{label}</p>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold text-gray-900">
            {typeof value === 'number' ? Math.round(value) : value}
          </span>
          {unit && <span className="text-sm text-gray-500">{unit}</span>}
        </div>
      </div>
    </div>
  );
};

const BatteryIndicator: React.FC<{ percentage: number }> = ({ percentage }) => {
  const getBatteryColor = (pct: number) => {
    if (pct > 50) return 'bg-green-500';
    if (pct > 20) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getBatteryBgColor = (pct: number) => {
    if (pct > 50) return 'bg-green-100';
    if (pct > 20) return 'bg-yellow-100';
    return 'bg-red-100';
  };

  return (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-100 p-4 ${getBatteryBgColor(percentage)}`}>
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg ${getBatteryBgColor(percentage)}`}>
          <Battery className={`w-5 h-5 ${percentage > 50 ? 'text-green-600' : percentage > 20 ? 'text-yellow-600' : 'text-red-600'}`} />
        </div>
        <div className={`px-2 py-1 rounded-full text-xs font-medium ${
          percentage > 50 ? 'bg-green-600 text-white' : 
          percentage > 20 ? 'bg-yellow-600 text-white' : 'bg-red-600 text-white'
        }`}>
          {percentage <= 20 ? 'LOW' : percentage <= 50 ? 'MID' : 'GOOD'}
        </div>
      </div>
      <div className="space-y-2">
        <p className="text-sm font-medium text-gray-600">Battery Level</p>
        <div className="flex items-baseline gap-1">
          <span className="text-2xl font-bold text-gray-900">{Math.round(percentage)}</span>
          <span className="text-sm text-gray-500">%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${getBatteryColor(percentage)}`}
            style={{ width: `${Math.max(0, Math.min(100, percentage))}%` }}
          ></div>
        </div>
      </div>
    </div>
  );
};

const formatTime = (seconds: number): string => {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  return `${minutes}m ${secs}s`;
};

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  batteryPercentage,
  motorSpeed,
  movementIntensity,
  totalDistance,
  sessionTime,
  faceTracking,
}) => {
  return (
    <div className="bg-gray-50 rounded-xl p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">System Metrics</h2>
        <p className="text-gray-600">Real-time wheelchair system status and performance</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Battery - Special component */}
        <BatteryIndicator percentage={batteryPercentage} />

        {/* Motor Speed */}
        <MetricCard
          icon={<Zap />}
          label="Motor Speed"
          value={motorSpeed}
          unit="%"
          color="blue"
        />

        {/* Movement Intensity */}
        <MetricCard
          icon={<Activity />}
          label="Movement Intensity"
          value={Math.round(movementIntensity * 100)}
          unit="%"
          color="purple"
        />

        {/* Session Time */}
        <MetricCard
          icon={<Clock />}
          label="Session Time"
          value={formatTime(sessionTime)}
          color="indigo"
        />

        {/* Total Distance */}
        <MetricCard
          icon={<MapPin />}
          label="Distance Traveled"
          value={totalDistance}
          unit="m"
          color="green"
        />

        {/* Face Tracking Status */}
        <MetricCard
          icon={<Eye />}
          label="Face Tracking"
          value={faceTracking ? "Active" : "Inactive"}
          color={faceTracking ? "green" : "red"}
          statusIndicator={true}
        />
      </div>

      {/* Status Summary */}
      <div className="mt-6 p-4 bg-white rounded-lg border border-gray-100">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-600">System Status:</span>
          <span className={`font-medium ${
            batteryPercentage > 20 && faceTracking ? 'text-green-600' : 'text-yellow-600'
          }`}>
            {batteryPercentage > 20 && faceTracking ? 'Operational' : 'Warning'}
          </span>
        </div>
      </div>
    </div>
  );
};