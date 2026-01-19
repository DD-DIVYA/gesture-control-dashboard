import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  mode: 'STOP' | 'WHEELCHAIR' | 'PLACE';
  rooms: string[];
  highlight: string | null;
  selected: string | null;
  batteryPercentage: number;
  motorSpeed: number;
  movementIntensity: number;
  totalDistance: number;
  sessionTime: number;
  faceTracking: boolean;
}

export interface WebSocketMessage {
  event: string;
  payload?: any;
}

export const useWebSocket = (url: string) => {
  const ws = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout | null>(null);
  const reconnectDelay = useRef(2000); // 2 seconds like Python client
  
  const [state, setState] = useState<WebSocketState>({
    connected: false,
    connecting: false,
    mode: 'STOP',
    rooms: ["Kitchen", "Bedroom", "Living Room", "Restroom"],
    highlight: null,
    selected: null,
    batteryPercentage: 85.0,
    motorSpeed: 0.0,
    movementIntensity: 0.0,
    totalDistance: 0.0,
    sessionTime: 0,
    faceTracking: false,
  });

  const [lastHeadDirection, setLastHeadDirection] = useState<string>('STOP');
  const [notifications, setNotifications] = useState<Array<{ id: string; message: string; type: 'info' | 'error' | 'success' }>>([]);

  const addNotification = useCallback((message: string, type: 'info' | 'error' | 'success' = 'info') => {
    const id = Date.now().toString();
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 4000);
  }, []);

  const connect = useCallback(() => {
    if (ws.current?.readyState === WebSocket.OPEN || state.connecting) return;

    // Clean up existing connection
    if (ws.current) {
      ws.current.close();
    }

    setState(prev => ({ ...prev, connecting: true }));

    try {
      ws.current = new WebSocket(url);
      
      ws.current.onopen = () => {
        setState(prev => ({ ...prev, connected: true, connecting: false }));
        reconnectDelay.current = 2000; // Reset delay on successful connection
        addNotification('Connected to wheelchair controller', 'success');
        console.log('✅ Connected to WebSocket server');
        
        // Request current status from backend
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ event: 'GET_STATUS' }));
        }
      };

      ws.current.onmessage = (event) => {
        try {
          console.log('📩 Received:', event.data);
          const message: WebSocketMessage = JSON.parse(event.data);
          
          switch (message.event) {
            case 'INIT':
              setState(prev => ({
                ...prev,
                rooms: message.payload?.rooms || prev.rooms,
                highlight: message.payload?.highlight || prev.highlight,
                mode: message.payload?.mode || 'STOP',
                selected: null,
              }));
              break;

            case 'MODE_CHANGE':
              const newMode = message.payload?.mode || 'STOP';
              console.log('🔄 MODE_CHANGE:', newMode, 'Previous mode:', state.mode);
              setState(prev => ({ 
                ...prev, 
                mode: newMode,
                // Clear selections when entering STOP mode
                highlight: newMode === 'STOP' ? null : prev.highlight,
                selected: newMode === 'STOP' ? null : prev.selected,
              }));
              addNotification(`Mode changed to ${newMode}`, 'info');
              // Reset head direction when not in wheelchair mode
              if (newMode !== 'WHEELCHAIR') {
                setLastHeadDirection('STOP');
              }
              break;

            case 'HEAD_MOVE':
              // Always update head direction and metrics regardless of mode
              const direction = message.payload?.direction || 'STOP';
              const motorSpeed = message.payload?.motor_speed || 0;
              const batteryPercentage = message.payload?.battery_percentage || 0;
              
              console.log('🎯 HEAD_MOVE:', { direction, motorSpeed, batteryPercentage });
              
              setLastHeadDirection(direction);
              // Update movement metrics from HEAD_MOVE
              setState(prev => ({
                ...prev,
                motorSpeed: motorSpeed,
                movementIntensity: message.payload?.movement_intensity || 0,
                batteryPercentage: batteryPercentage,
                totalDistance: message.payload?.total_distance || prev.totalDistance,
                sessionTime: message.payload?.session_time || prev.sessionTime,
              }));
              break;

            case 'PLACE_HIGHLIGHT':
              const highlightedPlace = message.payload?.place || null;
              console.log('🏠 PLACE_HIGHLIGHT:', highlightedPlace, 'Current mode:', state.mode);
              setState(prev => ({
                ...prev,
                highlight: highlightedPlace,
              }));
              break;

            case 'PLACE_SELECT':
              const selectedPlace = message.payload?.place || null;
              console.log('✅ PLACE_SELECT:', selectedPlace, 'Current mode:', state.mode);
              setState(prev => ({ 
                ...prev, 
                selected: selectedPlace 
              }));
              addNotification(`Selected: ${selectedPlace}`, 'info');
              break;

            case 'SYSTEM_RESET':
              setState(prev => ({
                ...prev,
                mode: 'STOP',
                highlight: null,
                selected: null,
                motorSpeed: 0.0,
                movementIntensity: 0.0,
              }));
              setLastHeadDirection('STOP');
              addNotification('System reset to STOP mode', 'info');
              break;

            case 'SYSTEM_STATUS':
              // Handle periodic system status updates
              setState(prev => ({
                ...prev,
                batteryPercentage: message.payload?.battery_percentage || prev.batteryPercentage,
                motorSpeed: message.payload?.motor_speed || prev.motorSpeed,
                totalDistance: message.payload?.total_distance || prev.totalDistance,
                sessionTime: message.payload?.session_time || prev.sessionTime,
                faceTracking: message.payload?.face_tracking || false,
              }));
              break;

            case 'TRACKING':
              if (message.payload?.status === 'lost') {
                addNotification('Face tracking lost', 'error');
              }
              break;

            case 'CALIBRATED':
              addNotification('Head calibrated successfully', 'success');
              break;

            case 'CALIBRATED_EYES':
              addNotification('Eye baseline reset', 'success');
              break;

            case 'FACE_STATUS':
              // Handle face detection status from backend
              setState(prev => ({
                ...prev,
                faceTracking: message.payload?.active || false,
              }));
              console.log('👁️ Face detection:', message.payload);
              break;

            case 'BLINK_EVENT':
              // Handle blink event feedback
              const blinkMessage = message.payload?.message;
              const blinkType = message.payload?.type;
              if (blinkMessage) {
                let notificationType: 'info' | 'success' | 'error' = 'info';
                
                // Use different notification types for better visual feedback
                if (blinkType === 'long') {
                  notificationType = 'error'; // Red for reset/stop actions
                } else if (message.payload?.action === 'select_place') {
                  notificationType = 'success'; // Green for successful selections
                } else {
                  notificationType = 'info'; // Blue for mode changes and navigation
                }
                
                addNotification(blinkMessage, notificationType);
                console.log('👁️ Blink event:', message.payload);
              }
              break;

            case 'ERROR':
              addNotification(message.payload?.message || 'An error occurred', 'error');
              break;

            case 'COMMAND':
              console.log('Command:', message.payload?.text);
              break;

            default:
              console.log('Unknown event:', message.event);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      ws.current.onclose = (event) => {
        setState(prev => ({ ...prev, connected: false, connecting: false }));
        console.log(`🔌 WebSocket closed: code=${event.code} reason=${event.reason}`);
        
        // Only attempt reconnection if it wasn't a manual close
        if (event.code !== 1000) {
          addNotification('Connection lost. Attempting to reconnect...', 'error');
          
          if (reconnectTimeout.current) {
            clearTimeout(reconnectTimeout.current);
          }
          
          reconnectTimeout.current = setTimeout(() => {
            console.log('🔄 Attempting to reconnect...');
            connect();
          }, reconnectDelay.current);
        }
      };

      ws.current.onerror = (error) => {
        setState(prev => ({ ...prev, connecting: false }));
        console.error('❌ WebSocket error:', error);
        addNotification('WebSocket connection error', 'error');
      };

    } catch (error) {
      setState(prev => ({ ...prev, connecting: false }));
      addNotification('Failed to establish connection', 'error');
      console.error('❌ Failed to create WebSocket:', error);
    }
  }, [url, addNotification, state.connecting]);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      const messageStr = JSON.stringify(message);
      ws.current.send(messageStr);
      console.log('→ Sent:', messageStr);
    } else {
      addNotification('Cannot send message: Not connected', 'error');
    }
  }, [addNotification]);

  const disconnect = useCallback(() => {
    if (reconnectTimeout.current) {
      clearTimeout(reconnectTimeout.current);
      reconnectTimeout.current = null;
    }
    if (ws.current) {
      ws.current.close(1000); // Normal closure
    }
    setState(prev => ({ ...prev, connected: false, connecting: false }));
  }, []);

  const removeNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    state,
    lastHeadDirection,
    notifications,
    sendMessage,
    connect,
    disconnect,
    removeNotification,
  };
};