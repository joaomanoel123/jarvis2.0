/* ============================================
   JARVIS 2.0 - MAIN APPLICATION LOGIC
   ============================================ */

class JarvisApp {
    constructor() {
        this.wsClient = wsClient;
        this.scene = jarvisScene;
        this.uiState = {
            activePanel: 'system',
            isVoiceActive: false,
            isMenuOpen: false,
            systemMetrics: {
                cpu: 0,
                memory: 0,
                agents: 0,
                uptime: 0
            }
        };
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupWebSocketHandlers();
        this.setupGestureHandlers();
        this.startSystemMetricsUpdate();
        this.showNotification('Sistema JARVIS 2.0 inicializado', 'success');
    }

    setupEventListeners() {
        // Dock controls
        document.getElementById('voice-control').addEventListener('click', () => this.toggleVoiceControl());
        document.getElementById('gesture-control').addEventListener('click', () => this.toggleGestureControl());
        document.getElementById('menu-control').addEventListener('click', () => this.toggleMenu());
        document.getElementById('settings-control').addEventListener('click', () => this.openSettings());
        document.getElementById('help-control').addEventListener('click', () => this.openHelp());
        
        // Modal
        document.getElementById('send-command').addEventListener('click', () => this.sendCommand());
        document.getElementById('cancel-command').addEventListener('click', () => this.closeModal());
        document.getElementById('command-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendCommand();
        });
        
        document.querySelector('.modal-close').addEventListener('click', () => this.closeModal());
        
        // Fechar modal ao clicar fora
        document.getElementById('command-modal').addEventListener('click', (e) => {
            if (e.target.id === 'command-modal') this.closeModal();
        });
    }

    setupWebSocketHandlers() {
        // Conectado
        this.wsClient.onConnect = () => {
            this.updateConnectionStatus(true);
            this.showNotification('Conectado ao servidor', 'success');
        };
        
        // Desconectado
        this.wsClient.onDisconnect = () => {
            this.updateConnectionStatus(false);
            this.showNotification('Desconectado do servidor', 'error');
        };
        
        // Mensagem
        this.wsClient.onMessage = (data) => {
            console.log('Mensagem recebida:', data);
        };
        
        // Erro
        this.wsClient.onError = (error) => {
            console.error('Erro WebSocket:', error);
            this.showNotification('Erro de conexão', 'error');
        };
        
        // Eventos customizados
        document.addEventListener('ai-response', (e) => {
            this.handleAIResponse(e.detail);
        });
        
        document.addEventListener('system-status', (e) => {
            this.handleSystemStatus(e.detail);
        });
        
        document.addEventListener('agent-update', (e) => {
            this.handleAgentUpdate(e.detail);
        });
    }

    setupGestureHandlers() {
        document.addEventListener('gesture-swipe-left', () => {
            console.log('Gesto: Swipe Left');
            this.rotatePanel('left');
        });
        
        document.addEventListener('gesture-swipe-right', () => {
            console.log('Gesto: Swipe Right');
            this.rotatePanel('right');
        });
        
        document.addEventListener('gesture-open-hand', () => {
            console.log('Gesto: Open Hand');
            this.toggleMenu();
        });
        
        document.addEventListener('gesture-pinch', () => {
            console.log('Gesto: Pinch');
            this.activateCoreInteraction();
        });
        
        document.addEventListener('gesture-navigate-left', () => {
            this.navigatePanel('left');
        });
        
        document.addEventListener('gesture-navigate-right', () => {
            this.navigatePanel('right');
        });
    }

    startSystemMetricsUpdate() {
        setInterval(() => {
            this.updateSystemMetrics();
        }, 1000);
    }

    updateSystemMetrics() {
        // Simular métricas do sistema
        const cpu = Math.floor(Math.random() * 100);
        const memory = Math.floor(Math.random() * 100);
        const agents = Math.floor(Math.random() * 10);
        
        this.uiState.systemMetrics.cpu = cpu;
        this.uiState.systemMetrics.memory = memory;
        this.uiState.systemMetrics.agents = agents;
        this.uiState.systemMetrics.uptime++;
        
        // Atualizar UI
        document.getElementById('cpu-usage').textContent = cpu + '%';
        document.getElementById('memory-usage').textContent = memory + '%';
        document.getElementById('agent-count').textContent = agents;
        document.getElementById('uptime').textContent = this.formatUptime(this.uiState.systemMetrics.uptime);
        
        // Enviar métricas para backend
        if (this.wsClient && this.wsClient.connected) {
            this.wsClient.sendSystemMetrics({
                cpu: cpu,
                memory: memory,
                agents: agents,
                uptime: this.uiState.systemMetrics.uptime
            });
        }
    }

    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
    }

    updateConnectionStatus(connected) {
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (connected) {
            statusDot.style.backgroundColor = '#00ff88';
            statusDot.style.boxShadow = '0 0 10px #00ff88';
            statusText.textContent = 'CONECTADO';
        } else {
            statusDot.style.backgroundColor = '#ff0055';
            statusDot.style.boxShadow = '0 0 10px #ff0055';
            statusText.textContent = 'DESCONECTADO';
        }
    }

    toggleVoiceControl() {
        this.uiState.isVoiceActive = !this.uiState.isVoiceActive;
        
        if (this.uiState.isVoiceActive) {
            this.showNotification('Ativação por voz iniciada', 'success');
            this.scene.activateCore();
        } else {
            this.showNotification('Ativação por voz desativada', 'success');
            this.scene.deactivateCore();
        }
    }

    toggleGestureControl() {
        this.showNotification('Sistema de gestos ativo', 'success');
        this.scene.addParticleExplosion();
    }

    toggleMenu() {
        this.uiState.isMenuOpen = !this.uiState.isMenuOpen;
        const modal = document.getElementById('command-modal');
        
        if (this.uiState.isMenuOpen) {
            modal.classList.remove('hidden');
            document.getElementById('command-input').focus();
        } else {
            modal.classList.add('hidden');
        }
    }

    openSettings() {
        this.showNotification('Abrindo configurações...', 'success');
        this.addActivityLog('Configurações abertas');
    }

    openHelp() {
        this.showNotification('Ajuda: Use os painéis para monitorar o sistema', 'success');
    }

    sendCommand() {
        const input = document.getElementById('command-input');
        const command = input.value.trim();
        
        if (command) {
            this.wsClient.sendCommand(command);
            this.addActivityLog(`Comando enviado: ${command}`);
            this.addConversationHistory(`[${this.getCurrentTime()}] Você: ${command}`);
            input.value = '';
            this.closeModal();
            this.showNotification('Comando enviado', 'success');
        }
    }

    closeModal() {
        document.getElementById('command-modal').classList.add('hidden');
        this.uiState.isMenuOpen = false;
    }

    rotatePanel(direction) {
        const panels = ['system', 'communication', 'metrics'];
        const currentIndex = panels.indexOf(this.uiState.activePanel);
        let newIndex;
        
        if (direction === 'left') {
            newIndex = (currentIndex - 1 + panels.length) % panels.length;
        } else {
            newIndex = (currentIndex + 1) % panels.length;
        }
        
        this.uiState.activePanel = panels[newIndex];
        this.showNotification(`Painel: ${this.uiState.activePanel}`, 'success');
    }

    navigatePanel(direction) {
        this.rotatePanel(direction);
    }

    activateCoreInteraction() {
        this.showNotification('Núcleo ativado', 'success');
        this.scene.activateCore();
        
        setTimeout(() => {
            this.scene.deactivateCore();
        }, 2000);
    }

    handleAIResponse(content) {
        const responseDiv = document.getElementById('ai-response');
        responseDiv.innerHTML = `<p>${content}</p>`;
        this.addConversationHistory(`[${this.getCurrentTime()}] JARVIS: ${content}`);
    }

    handleSystemStatus(content) {
        console.log('Status do sistema:', content);
        this.addActivityLog(`Status: ${JSON.stringify(content)}`);
    }

    handleAgentUpdate(content) {
        console.log('Atualização de agente:', content);
        this.addActivityLog(`Agente atualizado: ${content.name}`);
    }

    addActivityLog(message) {
        const activityLog = document.getElementById('activity-log');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = `[${this.getCurrentTime()}] ${message}`;
        activityLog.insertBefore(entry, activityLog.firstChild);
        
        // Manter apenas os últimos 10 logs
        while (activityLog.children.length > 10) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }

    addConversationHistory(message) {
        const history = document.getElementById('conversation-history');
        const item = document.createElement('div');
        item.className = 'conversation-item';
        
        const parts = message.split(': ');
        if (parts.length === 2) {
            item.innerHTML = `<span class="timestamp">${parts[0]}</span><span class="message">${parts[1]}</span>`;
        } else {
            item.textContent = message;
        }
        
        history.insertBefore(item, history.firstChild);
        
        // Manter apenas os últimos 15 itens
        while (history.children.length > 15) {
            history.removeChild(history.lastChild);
        }
    }

    addTaskItem(taskName, status = 'active') {
        const taskList = document.getElementById('task-list');
        const item = document.createElement('div');
        item.className = 'task-item';
        item.innerHTML = `<span class="task-status">●</span><span class="task-name">${taskName}</span>`;
        taskList.appendChild(item);
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        container.appendChild(notification);
        
        // Remover após 3 segundos
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(50px)';
            setTimeout(() => {
                container.removeChild(notification);
            }, 300);
        }, 3000);
    }

    getCurrentTime() {
        const now = new Date();
        return `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}`;
    }
}

/* ============================================
   INICIALIZAÇÃO DA APLICAÇÃO
   ============================================ */

let jarvisApp = null;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        jarvisApp = new JarvisApp();
    });
} else {
    jarvisApp = new JarvisApp();
}

/* ============================================
   LIMPEZA AO DESCARREGAR
   ============================================ */

window.addEventListener('beforeunload', () => {
    if (jarvisScene) {
        jarvisScene.dispose();
    }
    if (wsClient) {
        wsClient.disconnect();
    }
});
