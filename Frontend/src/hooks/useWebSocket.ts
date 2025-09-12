import { useEffect, useRef, useState, useCallback } from 'react';

export interface WebSocketState {
  connected: boolean;
  connecting: boolean;
  mode: 'wheelchair' | 'place';
  rooms: string[];
  highlight: string | null;
  selected: string | null;
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
    mode: 'wheelchair',
    rooms: ["Kitchen", "Bedroom", "Living Room", "Restroom"],
    highlight: "Kitchen",
    selected: null,
  });

  const [lastHeadDirection, setLastHeadDirection] = useState<string>('stop');
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
                mode: message.payload?.mode || 'wheelchair',
                selected: null,
              }));
              break;

            case 'MODE_CHANGE':
              setState(prev => ({ ...prev, mode: message.payload?.mode || 'wheelchair' }));
              addNotification(`Mode changed to ${message.payload?.mode}`, 'info');
              break;

            case 'HEAD_MOVE':
              setLastHeadDirection(message.payload?.direction || 'stop');
              break;

            case 'PLACE_HIGHLIGHT':
              setState(prev => ({
                ...prev,
                highlight: message.payload?.room || null,
                selected: null,
              }));
              break;

            case 'PLACE_SELECT':
              setState(prev => ({ ...prev, selected: message.payload?.room || null }));
              addNotification(`Selected: ${message.payload?.room}`, 'info');
              break;

            case 'PLACE_GO':
              addNotification(`Heading to ${message.payload?.room}...`, 'success');
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