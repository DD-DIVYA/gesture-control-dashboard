import React from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

interface Notification {
  id: string;
  message: string;
  type: 'info' | 'error' | 'success';
}

interface NotificationToastProps {
  notifications: Notification[];
  onRemove?: (id: string) => void;
}

export const NotificationToast: React.FC<NotificationToastProps> = ({ notifications, onRemove }) => {
  const getIcon = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return <CheckCircle2 className="w-5 h-5" />;
      case 'error':
        return <AlertCircle className="w-5 h-5" />;
      default:
        return <Info className="w-5 h-5" />;
    }
  };

  const getStyles = (type: Notification['type']) => {
    switch (type) {
      case 'success':
        return 'bg-green-50 border-green-200 text-green-800';
      case 'error':
        return 'bg-red-50 border-red-200 text-red-800';
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800';
    }
  };

  if (notifications.length === 0) return null;

  // Limit to 3 most recent notifications
  const displayNotifications = notifications.slice(-3);

  return (
    <div className="fixed top-4 right-4 z-50 space-y-2 max-w-md">
      {displayNotifications.map((notification, index) => (
        <div
          key={notification.id}
          className={`
            flex items-center gap-3 p-4 rounded-lg border shadow-lg
            transform transition-all duration-300 ease-out
            animate-in slide-in-from-right-full
            ${getStyles(notification.type)}
          `}
          style={{
            animationDelay: `${index * 100}ms`
          }}
          role="alert"
          aria-live="polite"
        >
          <div className="flex-shrink-0">
            {getIcon(notification.type)}
          </div>
          <p className="text-sm font-medium flex-1">
            {notification.message}
          </p>
          {onRemove && (
            <button
              onClick={() => onRemove(notification.id)}
              className="flex-shrink-0 p-1 rounded-full hover:bg-black/10 transition-colors"
              aria-label="Dismiss notification"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      ))}
    </div>
  );
};