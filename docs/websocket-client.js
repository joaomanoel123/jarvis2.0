/* ============================================
   JARVIS 2.0 - WEBSOCKET CLIENT
   ============================================ */

class WebSocketClient {
    constructor(url = 'ws://localhost:8000/ws') {
        this.url = url;
        this.ws = null;
        this.connected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        
        // Callbacks
        this.onConnect = null;
        this.onDisconnect = null;
        this.onMessage = null;
        this.onError = null;
        
        // Gestos suportados
        this.gestureHandlers = {};
        this.setupGestureHandlers();
        
        // Iniciar conexão
        this.connect();
    }

    connect() {
        try {
            this.ws = new WebSocket(this.url);
            
            this.ws.onopen = () => {
                this.connected = true;
                this.reconnectAttempts = 0;
                console.log('WebSocket conectado');
                
                if (this.onConnect) {
                    this.onConnect();
                }
                
                // Enviar mensagem de inicialização
                this.send({
                    type: 'init',
                    client: 'jarvis-frontend',
                    timestamp: new Date().toISOString()
                });
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                    
                    if (this.onMessage) {
                        this.onMessage(data);
                    }
                } catch (error) {
                    console.error('Erro ao parsear mensagem WebSocket:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('Erro WebSocket:', error);
                
                if (this.onError) {
                    this.onError(error);
                }
            };
            
            this.ws.onclose = () => {
                this.connected = false;
                console.log('WebSocket desconectado');
                
                if (this.onDisconnect) {
                    this.onDisconnect();
                }
                
                // Tentar reconectar
                this.attemptReconnect();
            };
        } catch (error) {
            console.error('Erro ao conectar WebSocket:', error);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Tentando reconectar... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            
            setTimeout(() => {
                this.connect();
            }, this.reconnectDelay);
        } else {
            console.error('Falha ao reconectar após múltiplas tentativas');
        }
    }

    send(data) {
        if (this.connected && this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.ws.send(JSON.stringify(data));
            } catch (error) {
                console.error('Erro ao enviar mensagem WebSocket:', error);
            }
        } else {
            console.warn('WebSocket não está conectado');
        }
    }

    handleMessage(data) {
        const { type, content, command } = data;
        
        switch (type) {
            case 'ai_response':
                this.handleAIResponse(content);
                break;
            
            case 'gesture_command':
                this.handleGestureCommand(command);
                break;
            
            case 'system_status':
                this.handleSystemStatus(content);
                break;
            
            case 'agent_update':
                this.handleAgentUpdate(content);
                break;
            
            case 'error':
                this.handleError(content);
                break;
            
            default:
                console.log('Tipo de mensagem desconhecido:', type);
        }
    }

    handleAIResponse(content) {
        // Disparar evento customizado
        const event = new CustomEvent('ai-response', { detail: content });
        document.dispatchEvent(event);
    }

    handleGestureCommand(command) {
        console.log('Comando de gesto recebido:', command);
        
        if (this.gestureHandlers[command]) {
            this.gestureHandlers[command]();
        } else {
            console.warn('Gesto não reconhecido:', command);
        }
    }

    handleSystemStatus(content) {
        const event = new CustomEvent('system-status', { detail: content });
        document.dispatchEvent(event);
    }

    handleAgentUpdate(content) {
        const event = new CustomEvent('agent-update', { detail: content });
        document.dispatchEvent(event);
    }

    handleError(content) {
        const event = new CustomEvent('ws-error', { detail: content });
        document.dispatchEvent(event);
    }

    setupGestureHandlers() {
        this.gestureHandlers = {
            'SWIPE_LEFT': () => {
                console.log('Gesto: Deslizar para esquerda');
                const event = new CustomEvent('gesture-swipe-left');
                document.dispatchEvent(event);
            },
            
            'SWIPE_RIGHT': () => {
                console.log('Gesto: Deslizar para direita');
                const event = new CustomEvent('gesture-swipe-right');
                document.dispatchEvent(event);
            },
            
            'SWIPE_UP': () => {
                console.log('Gesto: Deslizar para cima');
                const event = new CustomEvent('gesture-swipe-up');
                document.dispatchEvent(event);
            },
            
            'SWIPE_DOWN': () => {
                console.log('Gesto: Deslizar para baixo');
                const event = new CustomEvent('gesture-swipe-down');
                document.dispatchEvent(event);
            },
            
            'OPEN_HAND': () => {
                console.log('Gesto: Mão aberta');
                const event = new CustomEvent('gesture-open-hand');
                document.dispatchEvent(event);
            },
            
            'CLOSED_HAND': () => {
                console.log('Gesto: Mão fechada');
                const event = new CustomEvent('gesture-closed-hand');
                document.dispatchEvent(event);
            },
            
            'PINCH': () => {
                console.log('Gesto: Pinça');
                const event = new CustomEvent('gesture-pinch');
                document.dispatchEvent(event);
            },
            
            'POINT': () => {
                console.log('Gesto: Apontar');
                const event = new CustomEvent('gesture-point');
                document.dispatchEvent(event);
            },
            
            'NAVIGATE_LEFT': () => {
                console.log('Gesto: Navegar esquerda');
                const event = new CustomEvent('gesture-navigate-left');
                document.dispatchEvent(event);
            },
            
            'NAVIGATE_RIGHT': () => {
                console.log('Gesto: Navegar direita');
                const event = new CustomEvent('gesture-navigate-right');
                document.dispatchEvent(event);
            }
        };
    }

    // Métodos públicos para enviar dados
    sendCommand(command, params = {}) {
        this.send({
            type: 'command',
            command: command,
            params: params,
            timestamp: new Date().toISOString()
        });
    }

    sendQuery(query) {
        this.send({
            type: 'query',
            content: query,
            timestamp: new Date().toISOString()
        });
    }

    sendGestureRecognition(gesture, confidence = 1.0) {
        this.send({
            type: 'gesture_recognition',
            gesture: gesture,
            confidence: confidence,
            timestamp: new Date().toISOString()
        });
    }

    sendSystemMetrics(metrics) {
        this.send({
            type: 'system_metrics',
            metrics: metrics,
            timestamp: new Date().toISOString()
        });
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

/* ============================================
   GESTURE RECOGNITION SIMULATOR
   ============================================ */

class GestureRecognizer {
    constructor(wsClient) {
        this.wsClient = wsClient;
        this.touchStartX = 0;
        this.touchStartY = 0;
        this.touchEndX = 0;
        this.touchEndY = 0;
        this.minSwipeDistance = 50;
        
        this.setupTouchListeners();
        this.setupKeyboardShortcuts();
    }

    setupTouchListeners() {
        document.addEventListener('touchstart', (e) => {
            this.touchStartX = e.changedTouches[0].screenX;
            this.touchStartY = e.changedTouches[0].screenY;
        }, false);

        document.addEventListener('touchend', (e) => {
            this.touchEndX = e.changedTouches[0].screenX;
            this.touchEndY = e.changedTouches[0].screenY;
            this.handleSwipe();
        }, false);
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Atalhos para simular gestos
            switch (e.key.toLowerCase()) {
                case 'arrowleft':
                    this.wsClient.sendGestureRecognition('SWIPE_LEFT');
                    break;
                case 'arrowright':
                    this.wsClient.sendGestureRecognition('SWIPE_RIGHT');
                    break;
                case 'arrowup':
                    this.wsClient.sendGestureRecognition('SWIPE_UP');
                    break;
                case 'arrowdown':
                    this.wsClient.sendGestureRecognition('SWIPE_DOWN');
                    break;
                case 'o':
                    this.wsClient.sendGestureRecognition('OPEN_HAND');
                    break;
                case 'c':
                    this.wsClient.sendGestureRecognition('CLOSED_HAND');
                    break;
                case 'p':
                    this.wsClient.sendGestureRecognition('PINCH');
                    break;
            }
        });
    }

    handleSwipe() {
        const diffX = this.touchStartX - this.touchEndX;
        const diffY = this.touchStartY - this.touchEndY;

        if (Math.abs(diffX) > Math.abs(diffY)) {
            // Movimento horizontal
            if (Math.abs(diffX) > this.minSwipeDistance) {
                if (diffX > 0) {
                    this.wsClient.sendGestureRecognition('SWIPE_LEFT');
                } else {
                    this.wsClient.sendGestureRecognition('SWIPE_RIGHT');
                }
            }
        } else {
            // Movimento vertical
            if (Math.abs(diffY) > this.minSwipeDistance) {
                if (diffY > 0) {
                    this.wsClient.sendGestureRecognition('SWIPE_UP');
                } else {
                    this.wsClient.sendGestureRecognition('SWIPE_DOWN');
                }
            }
        }
    }
}

/* ============================================
   INICIALIZAÇÃO GLOBAL
   ============================================ */

let wsClient = null;
let gestureRecognizer = null;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        wsClient = new WebSocketClient();
        gestureRecognizer = new GestureRecognizer(wsClient);
    });
} else {
    wsClient = new WebSocketClient();
    gestureRecognizer = new GestureRecognizer(wsClient);
}
