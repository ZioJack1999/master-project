/**
 * VirtualHand3D - Real-Time 3D Biomechanical Virtual Hand Engine
 * 
 * Renders an articulated 19-joint cyber-robotic hand in WebGL (Three.js).
 * Animates 60 FPS continuous anatomical kinematics for Rest, Volar Grasp, and Finger Extension.
 * Features glowing cyber-tendons, dynamic joint pulsation, customizable themes, and full camera control.
 */

class VirtualHand3D {
  constructor(containerId, options = {}) {
    this.container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!this.container) {
      console.warn('[VirtualHand3D] Container element not found:', containerId);
      return;
    }

    this.options = Object.assign({
      theme: 'cyber', // 'cyber', 'bionic', 'matrix', 'stealth'
      antialias: true,
      fov: 42,
      cameraPos: [0, 0.4, 2.6],
      cameraTarget: [0, 0.25, 0],
      autoRotate: false
    }, options);

    // Target and current kinematics (smoothed via exponential damping)
    this.target = {
      flexion: 0.12,   // 0.0 (straight) to 1.0 (tight fist); 0.12 = natural rest; negative = hyperextension
      splay: 0.0,      // 0.0 (neutral) to 1.0 (wide splay / abduction)
      confidence: 0.0,
      activeClass: 'rest'
    };
    this.current = {
      flexion: 0.12,
      splay: 0.0,
      confidence: 0.0
    };

    this.joints = {};
    this.tendons = [];
    this.materials = [];
    this.wireframe = false;
    this.currentTheme = this.options.theme || 'cyber';
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };
    this.spherical = { radius: 2.6, theta: 0.0, phi: Math.PI / 2.3 };

    this.initScene();
    this.buildBiomechanicalHand();
    this.initControls();
    this.setTheme(this.currentTheme);
    this.animate = this.animate.bind(this);
    this.clock = new THREE.Clock();
    requestAnimationFrame(this.animate);
  }

  initScene() {
    const width = this.container.clientWidth || 360;
    const height = this.container.clientHeight || 280;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a1018);
    this.scene.fog = new THREE.FogExp2(0x0a1018, 0.28);

    this.camera = new THREE.PerspectiveCamera(this.options.fov, width / height, 0.1, 50);
    this.updateCameraPosition();

    this.renderer = new THREE.WebGLRenderer({ antialias: this.options.antialias, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Studio Lighting
    this.ambientLight = new THREE.AmbientLight(0x283b54, 1.4);
    this.scene.add(this.ambientLight);

    this.keyLight = new THREE.DirectionalLight(0xa5d8ff, 2.4);
    this.keyLight.position.set(2, 4, 3);
    this.keyLight.castShadow = true;
    this.scene.add(this.keyLight);

    this.rimLight = new THREE.DirectionalLight(0x00e5ff, 2.0);
    this.rimLight.position.set(-3, 2, -2);
    this.scene.add(this.rimLight);

    this.fillLight = new THREE.PointLight(0x71c8f6, 1.2, 10);
    this.fillLight.position.set(0, -2, 2);
    this.scene.add(this.fillLight);

    // Subtle reflective grid floor
    this.gridHelper = new THREE.GridHelper(5, 20, 0x1a2e47, 0x0f1c2d);
    this.gridHelper.position.y = -0.65;
    this.scene.add(this.gridHelper);

    // Resize observer
    if (window.ResizeObserver) {
      new ResizeObserver(() => this.onResize()).observe(this.container);
    }
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  buildBiomechanicalHand() {
    this.handRoot = new THREE.Group();
    this.handRoot.position.set(0, -0.2, 0);
    this.scene.add(this.handRoot);

    // High-tech Biomechanical Materials
    this.metalMaterial = new THREE.MeshStandardMaterial({
      color: 0x3b4a5d,
      roughness: 0.28,
      metalness: 0.88,
      wireframe: false
    });

    this.jointMaterial = new THREE.MeshStandardMaterial({
      color: 0x0a1926,
      emissive: 0x00d4ff,
      emissiveIntensity: 0.55,
      roughness: 0.15,
      metalness: 0.95,
      wireframe: false
    });

    this.palmMaterial = new THREE.MeshStandardMaterial({
      color: 0x223042,
      roughness: 0.35,
      metalness: 0.82,
      wireframe: false
    });

    this.tendonMaterial = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.75,
      wireframe: false
    });

    this.materials = [this.metalMaterial, this.jointMaterial, this.palmMaterial, this.tendonMaterial];

    // Palm / Carpal Base
    const palmGeo = new THREE.BoxGeometry(0.56, 0.62, 0.14);
    const palm = new THREE.Mesh(palmGeo, this.palmMaterial);
    palm.castShadow = true;
    palm.receiveShadow = true;
    this.handRoot.add(palm);

    // Glowing core reactor in center of palm
    const coreGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.15, 16);
    coreGeo.rotateX(Math.PI / 2);
    const coreMesh = new THREE.Mesh(coreGeo, this.jointMaterial);
    palm.add(coreMesh);

    // Wrist / Forearm Connector Sleeve
    const wristGeo = new THREE.CylinderGeometry(0.24, 0.27, 0.22, 18);
    const wristMesh = new THREE.Mesh(wristGeo, this.metalMaterial);
    wristMesh.position.set(0, -0.42, 0);
    wristMesh.castShadow = true;
    wristMesh.receiveShadow = true;
    this.handRoot.add(wristMesh);

    const wristRingGeo = new THREE.TorusGeometry(0.25, 0.022, 8, 24);
    wristRingGeo.rotateX(Math.PI / 2);
    const wristRing = new THREE.Mesh(wristRingGeo, this.jointMaterial);
    wristRing.position.set(0, -0.42, 0);
    this.handRoot.add(wristRing);

    // Finger Definitions: [name, x_offset, scale, splayDir]
    const fingerDefs = [
      { name: 'index',  x: -0.20, y: 0.34, scale: 0.94, splayDir: -1.0 },
      { name: 'middle', x: -0.06, y: 0.36, scale: 1.00, splayDir: -0.2 },
      { name: 'ring',   x:  0.08, y: 0.34, scale: 0.92, splayDir:  0.6 },
      { name: 'pinky',  x:  0.21, y: 0.30, scale: 0.78, splayDir:  1.3 }
    ];

    fingerDefs.forEach(f => {
      this.buildFinger(f);
    });

    // Thumb (Opposable multi-axis joint)
    this.buildThumb();
  }

  buildFinger(def) {
    const scale = def.scale;
    const phalanxLens = [0.28 * scale, 0.22 * scale, 0.16 * scale];
    const phalanxRadius = 0.048 * scale;

    // MCP (Knuckle) Root
    const mcpPivot = new THREE.Group();
    mcpPivot.position.set(def.x, def.y, 0);
    this.handRoot.add(mcpPivot);

    const mcpKnuckle = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 1.15, 12, 12), this.jointMaterial);
    mcpPivot.add(mcpKnuckle);

    // Proximal Phalanx
    const pMesh = new THREE.Mesh(new THREE.CylinderGeometry(phalanxRadius * 0.9, phalanxRadius, phalanxLens[0], 10), this.metalMaterial);
    pMesh.position.y = phalanxLens[0] / 2;
    pMesh.castShadow = true;
    mcpPivot.add(pMesh);

    // PIP Joint
    const pipPivot = new THREE.Group();
    pipPivot.position.y = phalanxLens[0];
    mcpPivot.add(pipPivot);

    const pipKnuckle = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 0.95, 10, 10), this.jointMaterial);
    pipPivot.add(pipKnuckle);

    // Intermediate Phalanx
    const iMesh = new THREE.Mesh(new THREE.CylinderGeometry(phalanxRadius * 0.75, phalanxRadius * 0.9, phalanxLens[1], 10), this.metalMaterial);
    iMesh.position.y = phalanxLens[1] / 2;
    iMesh.castShadow = true;
    pipPivot.add(iMesh);

    // DIP Joint
    const dipPivot = new THREE.Group();
    dipPivot.position.y = phalanxLens[1];
    pipPivot.add(dipPivot);

    const dipKnuckle = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 0.8, 10, 10), this.jointMaterial);
    dipPivot.add(dipKnuckle);

    // Distal Phalanx & Tip
    const dMesh = new THREE.Mesh(new THREE.CylinderGeometry(phalanxRadius * 0.55, phalanxRadius * 0.75, phalanxLens[2], 10), this.metalMaterial);
    dMesh.position.y = phalanxLens[2] / 2;
    dMesh.castShadow = true;
    dipPivot.add(dMesh);

    const tipMesh = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 0.55, 10, 10), this.jointMaterial);
    tipMesh.position.y = phalanxLens[2];
    dipPivot.add(tipMesh);

    // Dorsal Cyber-Tendon Light Strips along phalanges
    const tendonGeo1 = new THREE.CylinderGeometry(0.008, 0.008, phalanxLens[0] * 0.85, 6);
    const tendon1 = new THREE.Mesh(tendonGeo1, this.tendonMaterial);
    tendon1.position.set(0, phalanxLens[0] / 2, phalanxRadius * 0.95);
    mcpPivot.add(tendon1);
    this.tendons.push(tendon1);

    const tendonGeo2 = new THREE.CylinderGeometry(0.006, 0.006, phalanxLens[1] * 0.85, 6);
    const tendon2 = new THREE.Mesh(tendonGeo2, this.tendonMaterial);
    tendon2.position.set(0, phalanxLens[1] / 2, phalanxRadius * 0.82);
    pipPivot.add(tendon2);
    this.tendons.push(tendon2);

    this.joints[def.name] = {
      mcp: mcpPivot,
      pip: pipPivot,
      dip: dipPivot,
      splayDir: def.splayDir,
      baseX: def.x
    };
  }

  buildThumb() {
    const scale = 0.88;
    const phalanxLens = [0.22 * scale, 0.20 * scale];
    const phalanxRadius = 0.056 * scale;

    // CMC Trapeziometacarpal root
    const cmcPivot = new THREE.Group();
    cmcPivot.position.set(-0.30, -0.06, 0.04);
    cmcPivot.rotation.set(0.35, 0.45, 0.65);
    this.handRoot.add(cmcPivot);

    const cmcMesh = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 1.25, 12, 12), this.jointMaterial);
    cmcPivot.add(cmcMesh);

    // Metacarpal shaft
    const metaMesh = new THREE.Mesh(new THREE.CylinderGeometry(phalanxRadius, phalanxRadius * 1.1, phalanxLens[0], 10), this.metalMaterial);
    metaMesh.position.y = phalanxLens[0] / 2;
    cmcPivot.add(metaMesh);

    // MCP
    const mcpPivot = new THREE.Group();
    mcpPivot.position.y = phalanxLens[0];
    cmcPivot.add(mcpPivot);

    const mcpMesh = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 1.05, 10, 10), this.jointMaterial);
    mcpPivot.add(mcpMesh);

    // Proximal
    const pMesh = new THREE.Mesh(new THREE.CylinderGeometry(phalanxRadius * 0.85, phalanxRadius, phalanxLens[1], 10), this.metalMaterial);
    pMesh.position.y = phalanxLens[1] / 2;
    mcpPivot.add(pMesh);

    // IP & Tip
    const ipPivot = new THREE.Group();
    ipPivot.position.y = phalanxLens[1];
    mcpPivot.add(ipPivot);

    const tipMesh = new THREE.Mesh(new THREE.SphereGeometry(phalanxRadius * 0.75, 10, 10), this.jointMaterial);
    tipMesh.position.y = 0.08;
    ipPivot.add(tipMesh);

    // Thumb Cyber-Tendon strip
    const thumbTendonGeo = new THREE.CylinderGeometry(0.008, 0.008, phalanxLens[0] * 0.85, 6);
    const thumbTendon = new THREE.Mesh(thumbTendonGeo, this.tendonMaterial);
    thumbTendon.position.set(0, phalanxLens[0] / 2, phalanxRadius * 0.95);
    cmcPivot.add(thumbTendon);
    this.tendons.push(thumbTendon);

    this.joints.thumb = {
      cmc: cmcPivot,
      mcp: mcpPivot,
      ip: ipPivot
    };
  }

  initControls() {
    const el = this.renderer.domElement;
    el.style.cursor = 'grab';

    // Mouse drag
    el.addEventListener('mousedown', (e) => {
      this.isDragging = true;
      el.style.cursor = 'grabbing';
      this.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    window.addEventListener('mouseup', () => {
      this.isDragging = false;
      if (el) el.style.cursor = 'grab';
    });

    window.addEventListener('mousemove', (e) => {
      if (!this.isDragging) return;
      const deltaX = e.clientX - this.previousMousePosition.x;
      const deltaY = e.clientY - this.previousMousePosition.y;

      this.spherical.theta -= deltaX * 0.008;
      this.spherical.phi = Math.max(0.2, Math.min(Math.PI - 0.2, this.spherical.phi - deltaY * 0.008));
      this.updateCameraPosition();

      this.previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    // Zoom wheel
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.spherical.radius = Math.max(1.3, Math.min(4.5, this.spherical.radius + e.deltaY * 0.002));
      this.updateCameraPosition();
    }, { passive: false });

    // Double-click resets camera
    el.addEventListener('dblclick', () => {
      this.resetCamera();
    });

    // Touch support for tablets / touch screens
    el.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this.isDragging = true;
        this.previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      }
    }, { passive: true });

    window.addEventListener('touchend', () => {
      this.isDragging = false;
    });

    window.addEventListener('touchmove', (e) => {
      if (!this.isDragging || e.touches.length !== 1) return;
      const deltaX = e.touches[0].clientX - this.previousMousePosition.x;
      const deltaY = e.touches[0].clientY - this.previousMousePosition.y;

      this.spherical.theta -= deltaX * 0.008;
      this.spherical.phi = Math.max(0.2, Math.min(Math.PI - 0.2, this.spherical.phi - deltaY * 0.008));
      this.updateCameraPosition();

      this.previousMousePosition = { x: e.touches[0].clientX, y: e.touches[0].clientY };
    }, { passive: true });
  }

  updateCameraPosition() {
    const r = this.spherical.radius;
    const phi = this.spherical.phi;
    const theta = this.spherical.theta;

    this.camera.position.x = r * Math.sin(phi) * Math.sin(theta);
    this.camera.position.y = r * Math.cos(phi) + 0.25;
    this.camera.position.z = r * Math.sin(phi) * Math.cos(theta);
    this.camera.lookAt(0, 0.22, 0);
  }

  resetCamera() {
    this.spherical = { radius: 2.6, theta: 0.0, phi: Math.PI / 2.3 };
    this.updateCameraPosition();
  }

  zoom(delta) {
    this.spherical.radius = Math.max(1.3, Math.min(4.5, this.spherical.radius + delta));
    this.updateCameraPosition();
  }

  setView(preset) {
    switch (preset) {
      case 'front':
        this.spherical.theta = 0.0;
        this.spherical.phi = Math.PI / 2.0;
        break;
      case 'oblique':
        this.spherical.theta = -Math.PI / 6.0;
        this.spherical.phi = Math.PI / 2.4;
        break;
      case 'side':
        this.spherical.theta = -Math.PI / 2.0;
        this.spherical.phi = Math.PI / 2.0;
        break;
      case 'top':
        this.spherical.theta = 0.0;
        this.spherical.phi = 0.28;
        break;
      default:
        this.resetCamera();
        return;
    }
    this.updateCameraPosition();
  }

  toggleAutoRotate() {
    this.options.autoRotate = !this.options.autoRotate;
    return this.options.autoRotate;
  }

  setWireframe(enabled) {
    this.wireframe = !!enabled;
    if (this.handRoot) {
      this.handRoot.traverse((child) => {
        if (child.isMesh && child.material) {
          if (Array.isArray(child.material)) {
            child.material.forEach(m => m.wireframe = this.wireframe);
          } else {
            child.material.wireframe = this.wireframe;
          }
        }
      });
    }
    return this.wireframe;
  }

  toggleWireframe() {
    return this.setWireframe(!this.wireframe);
  }

  setTheme(themeName) {
    this.currentTheme = themeName;
    const themes = {
      cyber: {
        background: 0x0a1018,
        fog: 0x0a1018,
        metal: 0x3b4a5d,
        palm: 0x223042,
        roughness: 0.28,
        metalness: 0.88,
        jointGlow: 0x00d4ff,
        tendon: 0x00e5ff,
        grid1: 0x1a2e47,
        grid2: 0x0f1c2d
      },
      bionic: {
        background: 0x0c131d,
        fog: 0x0c131d,
        metal: 0xc4cdd6,
        palm: 0x8a9ba8,
        roughness: 0.20,
        metalness: 0.92,
        jointGlow: 0xf3b653,
        tendon: 0xf3b653,
        grid1: 0x334455,
        grid2: 0x182430
      },
      matrix: {
        background: 0x050e09,
        fog: 0x050e09,
        metal: 0x1b3824,
        palm: 0x0f2416,
        roughness: 0.32,
        metalness: 0.75,
        jointGlow: 0x00ff88,
        tendon: 0x00ff88,
        grid1: 0x164228,
        grid2: 0x0a2214
      },
      stealth: {
        background: 0x060709,
        fog: 0x060709,
        metal: 0x202228,
        palm: 0x15171a,
        roughness: 0.45,
        metalness: 0.65,
        jointGlow: 0xff3b53,
        tendon: 0xff3b53,
        grid1: 0x3b1c23,
        grid2: 0x1a0d11
      }
    };

    const t = themes[themeName] || themes.cyber;
    if (this.scene) {
      this.scene.background.setHex(t.background);
      this.scene.fog.color.setHex(t.fog);
    }
    if (this.metalMaterial) {
      this.metalMaterial.color.setHex(t.metal);
      this.metalMaterial.roughness = t.roughness;
      this.metalMaterial.metalness = t.metalness;
    }
    if (this.palmMaterial) {
      this.palmMaterial.color.setHex(t.palm);
    }
    if (this.jointMaterial) {
      this.jointMaterial.emissive.setHex(t.jointGlow);
    }
    if (this.tendonMaterial) {
      this.tendonMaterial.color.setHex(t.tendon);
    }
    if (this.gridHelper && this.scene) {
      this.scene.remove(this.gridHelper);
      this.gridHelper = new THREE.GridHelper(5, 20, t.grid1, t.grid2);
      this.gridHelper.position.y = -0.65;
      this.scene.add(this.gridHelper);
    }
    // Re-apply wireframe if active
    if (this.wireframe) {
      this.setWireframe(true);
    }
    return this.currentTheme;
  }

  /**
   * Update the virtual hand target state from real-time BCI decoding.
   * @param {string} className 'rest', 'volar_grasp', 'finger_extension', 'right_hand_grasp', 'action', etc.
   * @param {object|number} probabilities Object with class probabilities (e.g. { rest: 0.1, volar_grasp: 0.85, ... }) or direct confidence float.
   * @param {number} continuousVal Continuous flexion score [-0.5 to 1.0]
   */
  setTargetState(className, probabilities = {}, continuousVal = null) {
    this.target.activeClass = className;
    let prob = 0.0;
    if (typeof probabilities === 'number') {
      prob = probabilities;
    } else if (probabilities && typeof probabilities[className] === 'number') {
      prob = probabilities[className];
    }
    this.target.confidence = Math.min(1.0, Math.max(0.0, prob));

    const isGrasp = (className === 'volar_grasp' || className === 'right_hand_grasp' || className === 'action');
    const isExtension = (className === 'finger_extension');

    if (isGrasp) {
      // Grasp: Flexion curls fingers smoothly into fist [0.0 -> 1.0] with opposable thumb
      const flex = continuousVal !== null ? continuousVal : Math.max(0.70, prob);
      this.target.flexion = Math.min(1.0, Math.max(0.0, flex));
      this.target.splay = 0.0;
    } else if (isExtension) {
      // Extension: Splay outward and hyperextend
      const ext = continuousVal !== null ? continuousVal : Math.max(0.70, prob);
      this.target.flexion = -0.32 * ext; // hyperextension backwards
      this.target.splay = Math.min(1.0, Math.max(0.0, ext));
    } else {
      // Rest: Natural resting open hand
      this.target.flexion = 0.12;
      this.target.splay = 0.0;
    }
  }

  getTelemetry() {
    return {
      activeClass: this.target.activeClass,
      confidence: this.current.confidence,
      targetConfidence: this.target.confidence,
      flexion: this.current.flexion,
      splay: this.current.splay,
      wireframe: !!this.wireframe,
      theme: this.currentTheme,
      autoRotate: !!this.options.autoRotate,
      camera: {
        radius: this.spherical.radius,
        theta: this.spherical.theta,
        phi: this.spherical.phi
      }
    };
  }

  animate() {
    requestAnimationFrame(this.animate);

    const dt = Math.min(this.clock.getDelta(), 0.08);

    // Auto rotate turntable if enabled
    if (this.options.autoRotate && !this.isDragging) {
      this.spherical.theta += dt * 0.35;
      this.updateCameraPosition();
    }

    // Exponential smoothing for natural bio-damping (180ms response)
    const smoothing = 1.0 - Math.exp(-14.0 * dt);

    this.current.flexion += (this.target.flexion - this.current.flexion) * smoothing;
    this.current.splay += (this.target.splay - this.current.splay) * smoothing;
    this.current.confidence += (this.target.confidence - this.current.confidence) * smoothing;

    const flex = this.current.flexion;
    const splay = this.current.splay;

    // Apply joint rotations across fingers
    const fingers = ['index', 'middle', 'ring', 'pinky'];
    fingers.forEach(name => {
      const f = this.joints[name];
      if (!f) return;

      if (flex >= 0) {
        // Volar Grasp flexion: MCP (up to 75 deg), PIP (up to 95 deg), DIP (up to 65 deg)
        f.mcp.rotation.x = flex * 1.30;
        f.pip.rotation.x = flex * 1.62;
        f.dip.rotation.x = flex * 1.05;
        // Convergence toward palm center
        f.mcp.rotation.z = (f.splayDir * -0.08) * flex;
      } else {
        // Hyperextension
        f.mcp.rotation.x = flex * 0.85; // flex is negative
        f.pip.rotation.x = 0;
        f.dip.rotation.x = 0;
        // Finger splay / abduction
        f.mcp.rotation.z = f.splayDir * (0.24 * splay);
      }
    });

    // Thumb kinematics
    const thumb = this.joints.thumb;
    if (thumb) {
      if (flex >= 0) {
        thumb.cmc.rotation.x = 0.35 + flex * 0.65;
        thumb.cmc.rotation.y = 0.45 + flex * 0.55;
        thumb.mcp.rotation.x = flex * 0.95;
        thumb.ip.rotation.x = flex * 1.15;
      } else {
        thumb.cmc.rotation.x = 0.35 - splay * 0.35;
        thumb.cmc.rotation.y = 0.45 - splay * 0.45;
        thumb.mcp.rotation.x = -splay * 0.25;
        thumb.ip.rotation.x = 0;
      }
    }

    // Dynamic glow on cyber tendons and joints based on active confidence
    const pulse = Math.sin(this.clock.getElapsedTime() * 4.0) * 0.15;
    const glow = Math.min(1.8, Math.max(0.35, this.current.confidence * 1.4 + pulse));

    if (this.jointMaterial) {
      if (this.target.activeClass === 'finger_extension') {
        this.jointMaterial.emissive.setHex(0xad97f6); // purple/cyan extension glow
      } else if (this.target.activeClass === 'volar_grasp' || this.target.activeClass === 'right_hand_grasp' || this.target.activeClass === 'action') {
        this.jointMaterial.emissive.setHex(0x00e5ff); // electric cyan grasp glow
      } else {
        // Calm resting state glow based on theme
        if (this.currentTheme === 'matrix') {
          this.jointMaterial.emissive.setHex(0x00ff88);
        } else if (this.currentTheme === 'stealth') {
          this.jointMaterial.emissive.setHex(0xff3b53);
        } else if (this.currentTheme === 'bionic') {
          this.jointMaterial.emissive.setHex(0xf3b653);
        } else {
          this.jointMaterial.emissive.setHex(0x3a6073);
        }
      }
      this.jointMaterial.emissiveIntensity = glow;
    }

    if (this.tendonMaterial) {
      if (this.target.activeClass === 'finger_extension') {
        this.tendonMaterial.color.setHex(0xad97f6);
      } else if (this.target.activeClass === 'volar_grasp' || this.target.activeClass === 'right_hand_grasp' || this.target.activeClass === 'action') {
        this.tendonMaterial.color.setHex(0x00e5ff);
      } else {
        if (this.currentTheme === 'matrix') {
          this.tendonMaterial.color.setHex(0x00ff88);
        } else if (this.currentTheme === 'stealth') {
          this.tendonMaterial.color.setHex(0xff3b53);
        } else if (this.currentTheme === 'bionic') {
          this.tendonMaterial.color.setHex(0xf3b653);
        } else {
          this.tendonMaterial.color.setHex(0x3a6073);
        }
      }
      this.tendonMaterial.opacity = 0.4 + Math.min(0.55, this.current.confidence * 0.5 + pulse * 0.2);
    }

    if (this.rimLight) {
      this.rimLight.intensity = 1.4 + this.current.confidence * 1.2;
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// Expose globally
window.VirtualHand3D = VirtualHand3D;
