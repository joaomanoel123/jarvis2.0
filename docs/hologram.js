/**
 * J.A.R.V.I.S Holographic Interface
 * Engine 3D com Three.js
 * 
 * Características:
 * - Cena 3D com câmera e renderer
 * - Anéis rotativos animados
 * - Núcleo holográfico pulsante
 * - Sistema de partículas
 * - Responsividade mobile
 */

class HologramEngine {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.particles = [];
        this.rings = [];
        this.core = null;
        this.isActive = false;
        this.animationId = null;
        this.time = 0;

        // Configurações
        this.config = {
            particleCount: 500,
            particleSize: 2,
            ringCount: 3,
            coreSize: 5,
            backgroundColor: 0x000000
        };

        this.init();
    }

    /**
     * Inicializar engine
     */
    init() {
        this.setupScene();
        this.setupCamera();
        this.setupRenderer();
        this.setupLighting();
        this.createGeometries();
        this.setupEventListeners();
        this.animate();
        this.setupUI();
    }

    /**
     * Configurar cena
     */
    setupScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.backgroundColor);
        this.scene.fog = new THREE.Fog(this.config.backgroundColor, 500, 1000);
    }

    /**
     * Configurar câmera
     */
    setupCamera() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        const aspect = width / height;

        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 10000);
        this.camera.position.z = 80;
        this.camera.position.y = 20;
        this.camera.lookAt(0, 0, 0);
    }

    /**
     * Configurar renderer
     */
    setupRenderer() {
        const canvas = document.getElementById('hologram-canvas');
        this.renderer = new THREE.WebGLRenderer({ 
            canvas, 
            antialias: true, 
            alpha: true,
            precision: 'highp'
        });

        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowShadowMap;

        // Redimensionar ao mudar tamanho da janela
        window.addEventListener('resize', () => this.onWindowResize());
    }

    /**
     * Configurar iluminação
     */
    setupLighting() {
        // Luz ambiente
        const ambientLight = new THREE.AmbientLight(0x00d9ff, 0.4);
        this.scene.add(ambientLight);

        // Luz direcional
        const directionalLight = new THREE.DirectionalLight(0x00d9ff, 0.8);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);

        // Luz pontual (glow)
        const pointLight = new THREE.PointLight(0x00ffff, 1, 200);
        pointLight.position.set(0, 0, 0);
        this.scene.add(pointLight);
    }

    /**
     * Criar geometrias
     */
    createGeometries() {
        this.createRings();
        this.createCore();
        this.createParticles();
    }

    /**
     * Criar anéis rotativos
     */
    createRings() {
        const radii = [20, 35, 50];
        const colors = [0x00d9ff, 0x00ffff, 0x0099ff];

        radii.forEach((radius, index) => {
            const geometry = new THREE.BufferGeometry();
            const points = [];

            // Criar círculo
            for (let i = 0; i <= 128; i++) {
                const angle = (i / 128) * Math.PI * 2;
                points.push(
                    new THREE.Vector3(
                        Math.cos(angle) * radius,
                        0,
                        Math.sin(angle) * radius
                    )
                );
            }

            geometry.setFromPoints(points);

            const material = new THREE.LineBasicMaterial({
                color: colors[index],
                linewidth: 2,
                fog: false
            });

            const ring = new THREE.Line(geometry, material);
            ring.rotation.x = Math.random() * Math.PI;
            ring.rotation.y = Math.random() * Math.PI;
            ring.userData = {
                rotationSpeed: 0.001 * (index + 1),
                rotationAxis: new THREE.Vector3(
                    Math.random(),
                    Math.random(),
                    Math.random()
                ).normalize()
            };

            this.scene.add(ring);
            this.rings.push(ring);
        });
    }

    /**
     * Criar núcleo holográfico
     */
    createCore() {
        // Esfera interna
        const coreGeometry = new THREE.IcosahedronGeometry(this.config.coreSize, 4);
        const coreMaterial = new THREE.MeshStandardMaterial({
            color: 0x00d9ff,
            emissive: 0x00d9ff,
            emissiveIntensity: 0.8,
            metalness: 0.8,
            roughness: 0.2,
            wireframe: false
        });

        this.core = new THREE.Mesh(coreGeometry, coreMaterial);
        this.core.castShadow = true;
        this.core.receiveShadow = true;
        this.core.userData = {
            pulseSpeed: 0.02,
            baseScale: 1
        };

        this.scene.add(this.core);

        // Wireframe externo
        const wireframeGeometry = new THREE.IcosahedronGeometry(this.config.coreSize + 2, 4);
        const wireframeMaterial = new THREE.LineBasicMaterial({
            color: 0x00ffff,
            linewidth: 1,
            fog: false
        });

        const wireframe = new THREE.LineSegments(wireframeGeometry, wireframeMaterial);
        wireframe.userData = { rotationSpeed: -0.015 };
        this.scene.add(wireframe);
        this.rings.push(wireframe);
    }

    /**
     * Criar sistema de partículas
     */
    createParticles() {
        const geometry = new THREE.BufferGeometry();
        const positions = [];
        const velocities = [];

        for (let i = 0; i < this.config.particleCount; i++) {
            // Posição aleatória
            positions.push(
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200,
                (Math.random() - 0.5) * 200
            );

            // Velocidade aleatória
            velocities.push(
                (Math.random() - 0.5) * 0.5,
                (Math.random() - 0.5) * 0.5,
                (Math.random() - 0.5) * 0.5
            );
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));

        const material = new THREE.PointsMaterial({
            color: 0x00d9ff,
            size: this.config.particleSize,
            sizeAttenuation: true,
            fog: false
        });

        const particleSystem = new THREE.Points(geometry, material);
        particleSystem.userData = {
            velocities: velocities,
            positions: positions
        };

        this.scene.add(particleSystem);
        this.particles.push(particleSystem);
    }

    /**
     * Loop de animação
     */
    animate = () => {
        this.animationId = requestAnimationFrame(this.animate);
        this.time += 0.016; // ~60 FPS

        // Animar anéis
        this.rings.forEach(ring => {
            if (ring.userData.rotationSpeed) {
                ring.rotation.x += ring.userData.rotationSpeed * 0.5;
                ring.rotation.y += ring.userData.rotationSpeed;
                ring.rotation.z += ring.userData.rotationSpeed * 0.3;
            }
        });

        // Animar núcleo
        if (this.core) {
            const pulseScale = 1 + Math.sin(this.time * this.core.userData.pulseSpeed) * 0.1;
            this.core.scale.set(pulseScale, pulseScale, pulseScale);
            this.core.rotation.x += 0.003;
            this.core.rotation.y += 0.005;
        }

        // Animar partículas
        this.updateParticles();

        // Renderizar
        this.renderer.render(this.scene, this.camera);

        // Atualizar FPS
        this.updateFPS();
    };

    /**
     * Atualizar partículas
     */
    updateParticles() {
        this.particles.forEach(particleSystem => {
            const positions = particleSystem.geometry.attributes.position.array;
            const velocities = particleSystem.userData.velocities;

            for (let i = 0; i < positions.length; i += 3) {
                const velocityIndex = i / 3;
                const vx = velocities[velocityIndex * 3];
                const vy = velocities[velocityIndex * 3 + 1];
                const vz = velocities[velocityIndex * 3 + 2];

                positions[i] += vx;
                positions[i + 1] += vy;
                positions[i + 2] += vz;

                // Wrap around
                if (Math.abs(positions[i]) > 100) velocities[velocityIndex * 3] *= -1;
                if (Math.abs(positions[i + 1]) > 100) velocities[velocityIndex * 3 + 1] *= -1;
                if (Math.abs(positions[i + 2]) > 100) velocities[velocityIndex * 3 + 2] *= -1;
            }

            particleSystem.geometry.attributes.position.needsUpdate = true;
        });
    }

    /**
     * Atualizar FPS
     */
    updateFPS() {
        if (Math.floor(this.time) % 1 === 0) {
            const fpsCounter = document.getElementById('fps-counter');
            if (fpsCounter) {
                const fps = Math.round(1 / (this.time - Math.floor(this.time)));
                fpsCounter.textContent = Math.min(fps, 60);
            }
        }
    }

    /**
     * Ativar núcleo (quando JARVIS está respondendo)
     */
    activateCore() {
        this.isActive = true;
        if (this.core) {
            this.core.userData.pulseSpeed = 0.05;
            this.core.material.emissiveIntensity = 1.2;
        }
        document.getElementById('status-indicator').textContent = 'ACTIVE';
    }

    /**
     * Desativar núcleo
     */
    deactivateCore() {
        this.isActive = false;
        if (this.core) {
            this.core.userData.pulseSpeed = 0.02;
            this.core.material.emissiveIntensity = 0.8;
        }
        document.getElementById('status-indicator').textContent = 'READY';
    }

    /**
     * Handle resize de janela
     */
    onWindowResize = () => {
        const width = window.innerWidth;
        const height = window.innerHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    };

    /**
     * Configurar event listeners
     */
    setupEventListeners() {
        const userInput = document.getElementById('user-input');
        const sendBtn = document.getElementById('send-btn');
        const voiceBtn = document.getElementById('voice-btn');
        const resetBtn = document.getElementById('reset-btn');
        const infoBtn = document.getElementById('info-btn');
        const infoModal = document.getElementById('info-modal');
        const modalClose = document.querySelector('.modal-close');

        // Enviar mensagem
        sendBtn?.addEventListener('click', () => this.handleSendMessage(userInput));
        userInput?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.handleSendMessage(userInput);
            }
        });

        // Voz
        voiceBtn?.addEventListener('click', () => this.handleVoice());

        // Reset
        resetBtn?.addEventListener('click', () => this.handleReset());

        // Info
        infoBtn?.addEventListener('click', () => {
            infoModal?.classList.remove('hidden');
        });

        modalClose?.addEventListener('click', () => {
            infoModal?.classList.add('hidden');
        });

        infoModal?.addEventListener('click', (e) => {
            if (e.target === infoModal) {
                infoModal.classList.add('hidden');
            }
        });
    }

    /**
     * Configurar UI
     */
    setupUI() {
        const particleCounter = document.getElementById('particle-counter');
        if (particleCounter) {
            particleCounter.textContent = this.config.particleCount;
        }
    }

    /**
     * Handle enviar mensagem
     */
    handleSendMessage(inputElement) {
        const message = inputElement?.value.trim();
        if (!message) return;

        // Ativar núcleo
        this.activateCore();

        // Simular resposta
        const responseContainer = document.getElementById('response-container');
        if (responseContainer) {
            responseContainer.innerHTML = `<p class="response-text">Processando: "${message}"...</p>`;
        }

        // Simular delay de resposta
        setTimeout(() => {
            const responses = [
                'Entendido. Processando sua solicitação.',
                'Confirmado. Analisando dados.',
                'Recebido. Executando operação.',
                'Afirmativo. Sistema pronto.',
                'Reconhecido. Iniciando procedimento.'
            ];

            const randomResponse = responses[Math.floor(Math.random() * responses.length)];
            if (responseContainer) {
                responseContainer.innerHTML = `<p class="response-text">${randomResponse}</p>`;
            }

            // Desativar núcleo após resposta
            setTimeout(() => {
                this.deactivateCore();
            }, 2000);
        }, 1500);

        // Limpar input
        if (inputElement) {
            inputElement.value = '';
        }
    }

    /**
     * Handle voz
     */
    handleVoice() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            alert('Speech Recognition não suportado neste navegador');
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'pt-BR';

        recognition.onstart = () => {
            this.activateCore();
            const responseContainer = document.getElementById('response-container');
            if (responseContainer) {
                responseContainer.innerHTML = '<p class="response-text">Escutando...</p>';
            }
        };

        recognition.onresult = (event) => {
            const transcript = Array.from(event.results)
                .map(result => result[0].transcript)
                .join('');

            const userInput = document.getElementById('user-input');
            if (userInput) {
                userInput.value = transcript;
                this.handleSendMessage(userInput);
            }
        };

        recognition.onerror = (event) => {
            console.error('Erro no reconhecimento de voz:', event.error);
            this.deactivateCore();
        };

        recognition.start();
    }

    /**
     * Handle reset
     */
    handleReset() {
        const userInput = document.getElementById('user-input');
        if (userInput) {
            userInput.value = '';
        }

        const responseContainer = document.getElementById('response-container');
        if (responseContainer) {
            responseContainer.innerHTML = '<p class="response-text">Aguardando entrada...</p>';
        }

        this.deactivateCore();
    }

    /**
     * Destruir engine
     */
    destroy() {
        cancelAnimationFrame(this.animationId);
        this.renderer?.dispose();
        window.removeEventListener('resize', this.onWindowResize);
    }
}

// Inicializar quando DOM estiver pronto
document.addEventListener('DOMContentLoaded', () => {
    window.hologramEngine = new HologramEngine();
});

// Limpar ao descarregar página
window.addEventListener('beforeunload', () => {
    if (window.hologramEngine) {
        window.hologramEngine.destroy();
    }
});
