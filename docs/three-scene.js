/* ============================================
   JARVIS 2.0 - THREE.JS 3D SCENE
   ============================================ */

class JarvisThreeScene {
    constructor() {
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.canvas = null;
        
        // Objetos 3D
        this.coreGroup = null;
        this.hudsGroup = null;
        this.ringsGroup = null;
        this.gridGroup = null;
        this.particlesGroup = null;
        
        // Animação
        this.animationId = null;
        this.time = 0;
        
        this.init();
    }

    init() {
        // Obter canvas
        this.canvas = document.getElementById('three-canvas');
        
        // Criar cena
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e27);
        this.scene.fog = new THREE.Fog(0x0a0e27, 100, 1000);
        
        // Criar câmera
        this.camera = new THREE.PerspectiveCamera(
            75,
            window.innerWidth / window.innerHeight,
            0.1,
            1000
        );
        this.camera.position.z = 50;
        
        // Criar renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFShadowShadowMap;
        
        // Adicionar iluminação
        this.setupLighting();
        
        // Criar grupos
        this.coreGroup = new THREE.Group();
        this.hudsGroup = new THREE.Group();
        this.ringsGroup = new THREE.Group();
        this.gridGroup = new THREE.Group();
        this.particlesGroup = new THREE.Group();
        
        this.scene.add(this.coreGroup);
        this.scene.add(this.hudsGroup);
        this.scene.add(this.ringsGroup);
        this.scene.add(this.gridGroup);
        this.scene.add(this.particlesGroup);
        
        // Criar elementos 3D
        this.createAICore();
        this.createHUDElements();
        this.createRotatingRings();
        this.createHolographicGrid();
        this.createParticles();
        
        // Iniciar animação
        this.animate();
        
        // Listener para resize
        window.addEventListener('resize', () => this.onWindowResize());
    }

    setupLighting() {
        // Luz ambiente
        const ambientLight = new THREE.AmbientLight(0x00d4ff, 0.3);
        this.scene.add(ambientLight);
        
        // Luz direcional
        const directionalLight = new THREE.DirectionalLight(0x0066ff, 0.5);
        directionalLight.position.set(50, 50, 50);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        this.scene.add(directionalLight);
        
        // Luz pontual - Azul Neon
        const pointLight1 = new THREE.PointLight(0x00d4ff, 1, 200);
        pointLight1.position.set(0, 0, 30);
        this.scene.add(pointLight1);
        
        // Luz pontual - Ciano
        const pointLight2 = new THREE.PointLight(0x00f0ff, 0.8, 150);
        pointLight2.position.set(-40, 40, 20);
        this.scene.add(pointLight2);
        
        // Luz pontual - Roxo
        const pointLight3 = new THREE.PointLight(0x8800ff, 0.6, 150);
        pointLight3.position.set(40, -40, 20);
        this.scene.add(pointLight3);
    }

    createAICore() {
        // Núcleo central - Esfera brilhante
        const coreGeometry = new THREE.IcosahedronGeometry(8, 4);
        const coreMaterial = new THREE.MeshPhongMaterial({
            color: 0x00d4ff,
            emissive: 0x00d4ff,
            emissiveIntensity: 0.5,
            shininess: 100,
            wireframe: false
        });
        
        const coreMesh = new THREE.Mesh(coreGeometry, coreMaterial);
        coreMesh.castShadow = true;
        coreMesh.receiveShadow = true;
        this.coreGroup.add(coreMesh);
        
        // Wireframe do núcleo
        const wireframeGeometry = new THREE.IcosahedronGeometry(8.2, 4);
        const wireframeMaterial = new THREE.MeshPhongMaterial({
            color: 0x00f0ff,
            emissive: 0x00f0ff,
            emissiveIntensity: 0.3,
            wireframe: true,
            transparent: true,
            opacity: 0.6
        });
        
        const wireframeMesh = new THREE.Mesh(wireframeGeometry, wireframeMaterial);
        this.coreGroup.add(wireframeMesh);
        
        // Armazenar para animação
        this.coreMesh = coreMesh;
        this.wireframeMesh = wireframeMesh;
    }

    createHUDElements() {
        // Criar 4 painéis HUD circulares
        const hudPositions = [
            { x: 0, y: 25, z: 0, rotation: 0 },
            { x: 25, y: 0, z: 0, rotation: Math.PI / 2 },
            { x: 0, y: -25, z: 0, rotation: Math.PI },
            { x: -25, y: 0, z: 0, rotation: -Math.PI / 2 }
        ];
        
        hudPositions.forEach((pos, index) => {
            const hudGroup = new THREE.Group();
            
            // Círculo externo
            const circleGeometry = new THREE.BufferGeometry();
            const circlePoints = [];
            for (let i = 0; i <= 64; i++) {
                const angle = (i / 64) * Math.PI * 2;
                circlePoints.push(
                    Math.cos(angle) * 6,
                    Math.sin(angle) * 6,
                    0
                );
            }
            circleGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(circlePoints), 3));
            
            const circleMaterial = new THREE.LineBasicMaterial({
                color: 0x00d4ff,
                linewidth: 2
            });
            
            const circle = new THREE.Line(circleGeometry, circleMaterial);
            hudGroup.add(circle);
            
            // Linhas radiais
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * Math.PI * 2;
                const lineGeometry = new THREE.BufferGeometry();
                lineGeometry.setAttribute('position', new THREE.BufferAttribute(
                    new Float32Array([
                        0, 0, 0,
                        Math.cos(angle) * 6,
                        Math.sin(angle) * 6,
                        0
                    ]),
                    3
                ));
                
                const line = new THREE.Line(lineGeometry, circleMaterial);
                hudGroup.add(line);
            }
            
            // Posicionar HUD
            hudGroup.position.set(pos.x, pos.y, pos.z);
            hudGroup.rotation.z = pos.rotation;
            
            // Armazenar para animação
            hudGroup.hudIndex = index;
            this.hudsGroup.add(hudGroup);
        });
    }

    createRotatingRings() {
        // Criar anéis rotativos
        const ringConfigs = [
            { radius: 30, color: 0x00d4ff, speed: 0.005 },
            { radius: 40, color: 0x00f0ff, speed: -0.003 },
            { radius: 50, color: 0x0066ff, speed: 0.004 }
        ];
        
        ringConfigs.forEach((config) => {
            const ringGeometry = new THREE.BufferGeometry();
            const ringPoints = [];
            
            for (let i = 0; i <= 128; i++) {
                const angle = (i / 128) * Math.PI * 2;
                ringPoints.push(
                    Math.cos(angle) * config.radius,
                    Math.sin(angle) * config.radius,
                    0
                );
            }
            
            ringGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ringPoints), 3));
            
            const ringMaterial = new THREE.LineBasicMaterial({
                color: config.color,
                linewidth: 1,
                transparent: true,
                opacity: 0.6
            });
            
            const ring = new THREE.Line(ringGeometry, ringMaterial);
            ring.rotationSpeed = config.speed;
            
            this.ringsGroup.add(ring);
        });
        
        // Anéis em diferentes planos
        const ring2Geometry = new THREE.BufferGeometry();
        const ring2Points = [];
        for (let i = 0; i <= 128; i++) {
            const angle = (i / 128) * Math.PI * 2;
            ring2Points.push(
                Math.cos(angle) * 35,
                0,
                Math.sin(angle) * 35
            );
        }
        ring2Geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ring2Points), 3));
        const ring2Material = new THREE.LineBasicMaterial({ color: 0x8800ff, opacity: 0.4, transparent: true });
        const ring2 = new THREE.Line(ring2Geometry, ring2Material);
        ring2.rotationSpeed = 0.002;
        this.ringsGroup.add(ring2);
    }

    createHolographicGrid() {
        const gridSize = 80;
        const gridDivisions = 20;
        const gridGeometry = new THREE.BufferGeometry();
        const gridPoints = [];
        
        // Linhas horizontais
        for (let i = -gridDivisions; i <= gridDivisions; i++) {
            const pos = (i / gridDivisions) * gridSize;
            gridPoints.push(-gridSize, 0, pos, gridSize, 0, pos);
        }
        
        // Linhas verticais
        for (let i = -gridDivisions; i <= gridDivisions; i++) {
            const pos = (i / gridDivisions) * gridSize;
            gridPoints.push(pos, 0, -gridSize, pos, 0, gridSize);
        }
        
        gridGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(gridPoints), 3));
        
        const gridMaterial = new THREE.LineBasicMaterial({
            color: 0x0066ff,
            transparent: true,
            opacity: 0.1,
            linewidth: 0.5
        });
        
        const grid = new THREE.LineSegments(gridGeometry, gridMaterial);
        grid.position.y = -30;
        this.gridGroup.add(grid);
    }

    createParticles() {
        const particleCount = 300;
        const particleGeometry = new THREE.BufferGeometry();
        const positions = new Float32Array(particleCount * 3);
        const colors = new Float32Array(particleCount * 3);
        
        for (let i = 0; i < particleCount * 3; i += 3) {
            // Posições aleatórias
            positions[i] = (Math.random() - 0.5) * 200;
            positions[i + 1] = (Math.random() - 0.5) * 200;
            positions[i + 2] = (Math.random() - 0.5) * 200;
            
            // Cores aleatórias (azul, ciano, roxo)
            const colorChoice = Math.random();
            if (colorChoice < 0.33) {
                colors[i] = 0;
                colors[i + 1] = 0.83;
                colors[i + 2] = 1;
            } else if (colorChoice < 0.66) {
                colors[i] = 0;
                colors[i + 1] = 0.94;
                colors[i + 2] = 1;
            } else {
                colors[i] = 0.53;
                colors[i + 1] = 0;
                colors[i + 2] = 1;
            }
        }
        
        particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        particleGeometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        
        const particleMaterial = new THREE.PointsMaterial({
            size: 0.5,
            vertexColors: true,
            transparent: true,
            opacity: 0.6,
            sizeAttenuation: true
        });
        
        const particles = new THREE.Points(particleGeometry, particleMaterial);
        this.particlesGroup.add(particles);
        
        // Armazenar para animação
        this.particles = particles;
        this.particlePositions = positions;
    }

    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        this.time += 0.01;
        
        // Animar núcleo
        if (this.coreMesh) {
            this.coreMesh.rotation.x += 0.002;
            this.coreMesh.rotation.y += 0.003;
            
            // Pulsação
            const pulse = 1 + Math.sin(this.time * 2) * 0.1;
            this.coreMesh.scale.set(pulse, pulse, pulse);
        }
        
        if (this.wireframeMesh) {
            this.wireframeMesh.rotation.x -= 0.001;
            this.wireframeMesh.rotation.y -= 0.002;
        }
        
        // Animar HUDs
        this.hudsGroup.children.forEach((hud, index) => {
            hud.rotation.z += 0.01 * (index % 2 === 0 ? 1 : -1);
            
            // Movimento orbital
            const orbitRadius = 25;
            const orbitSpeed = 0.005;
            hud.position.x = Math.cos(this.time * orbitSpeed + index * Math.PI / 2) * orbitRadius;
            hud.position.y = Math.sin(this.time * orbitSpeed + index * Math.PI / 2) * orbitRadius;
        });
        
        // Animar anéis
        this.ringsGroup.children.forEach((ring) => {
            if (ring.rotationSpeed) {
                ring.rotation.z += ring.rotationSpeed;
            }
        });
        
        // Animar partículas
        if (this.particles) {
            const positions = this.particles.geometry.attributes.position.array;
            for (let i = 0; i < positions.length; i += 3) {
                positions[i] += Math.sin(this.time * 0.5 + i) * 0.1;
                positions[i + 1] += Math.cos(this.time * 0.5 + i) * 0.1;
                positions[i + 2] += Math.sin(this.time * 0.3 + i) * 0.05;
            }
            this.particles.geometry.attributes.position.needsUpdate = true;
        }
        
        // Renderizar
        this.renderer.render(this.scene, this.camera);
    }

    onWindowResize() {
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        
        this.renderer.setSize(width, height);
    }

    // Métodos públicos para interação
    activateCore() {
        if (this.coreMesh) {
            this.coreMesh.material.emissiveIntensity = 1;
        }
    }

    deactivateCore() {
        if (this.coreMesh) {
            this.coreMesh.material.emissiveIntensity = 0.5;
        }
    }

    changeGridColor(color) {
        this.gridGroup.children.forEach(grid => {
            if (grid.material) {
                grid.material.color.setHex(color);
            }
        });
    }

    addParticleExplosion(position = { x: 0, y: 0, z: 0 }) {
        const explosionCount = 50;
        const explosionGeometry = new THREE.BufferGeometry();
        const explosionPositions = new Float32Array(explosionCount * 3);
        
        for (let i = 0; i < explosionCount * 3; i += 3) {
            explosionPositions[i] = position.x + (Math.random() - 0.5) * 10;
            explosionPositions[i + 1] = position.y + (Math.random() - 0.5) * 10;
            explosionPositions[i + 2] = position.z + (Math.random() - 0.5) * 10;
        }
        
        explosionGeometry.setAttribute('position', new THREE.BufferAttribute(explosionPositions, 3));
        
        const explosionMaterial = new THREE.PointsMaterial({
            size: 1,
            color: 0x00f0ff,
            transparent: true,
            opacity: 0.8
        });
        
        const explosion = new THREE.Points(explosionGeometry, explosionMaterial);
        this.scene.add(explosion);
        
        // Remover após animação
        setTimeout(() => {
            this.scene.remove(explosion);
        }, 1000);
    }

    dispose() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        this.renderer.dispose();
    }
}

// Inicializar cena quando o script carregar
let jarvisScene = null;

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        jarvisScene = new JarvisThreeScene();
    });
} else {
    jarvisScene = new JarvisThreeScene();
}
