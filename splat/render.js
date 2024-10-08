
let camera = {
    width: 1024,
    height: 1024,
    position: [0, 0, 0.8],
    rotation: [[ 1, 0, 0], [0, -1, 0], [ 0, 0, -1]],
    fy: 1024,
    fx: 1024,
};

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function getViewMatrix(camera) {
    const R = camera.rotation.flat();
    const t = camera.position;
    const camToWorld = [
        [R[0], R[1], R[2], 0],
        [R[3], R[4], R[5], 0],
        [R[6], R[7], R[8], 0],
        [
            -t[0] * R[0] - t[1] * R[3] - t[2] * R[6],
            -t[0] * R[1] - t[1] * R[4] - t[2] * R[7],
            -t[0] * R[2] - t[1] * R[5] - t[2] * R[8],
            1,
        ],
    ].flat();
    return camToWorld;
}

function multiply4(a, b) {
    return [
        b[0] * a[0] + b[1] * a[4] + b[2] * a[8] + b[3] * a[12],
        b[0] * a[1] + b[1] * a[5] + b[2] * a[9] + b[3] * a[13],
        b[0] * a[2] + b[1] * a[6] + b[2] * a[10] + b[3] * a[14],
        b[0] * a[3] + b[1] * a[7] + b[2] * a[11] + b[3] * a[15],
        b[4] * a[0] + b[5] * a[4] + b[6] * a[8] + b[7] * a[12],
        b[4] * a[1] + b[5] * a[5] + b[6] * a[9] + b[7] * a[13],
        b[4] * a[2] + b[5] * a[6] + b[6] * a[10] + b[7] * a[14],
        b[4] * a[3] + b[5] * a[7] + b[6] * a[11] + b[7] * a[15],
        b[8] * a[0] + b[9] * a[4] + b[10] * a[8] + b[11] * a[12],
        b[8] * a[1] + b[9] * a[5] + b[10] * a[9] + b[11] * a[13],
        b[8] * a[2] + b[9] * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8] * a[3] + b[9] * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8] + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9] + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
    ];
}

function createWorker(self) {
    let buffer;
    let vertexCount = 0;
    let viewProj;
    // 6*4 + 4 + 4 = 8*4
    // XYZ - Position (Float32)
    // XYZ - Scale (Float32)
    // RGBA - colors (uint8)
    // IJKL - quaternion/rot (uint8)
    let lastProj = [];
    let depthIndex = new Uint32Array();
    let lastVertexCount = 0;

    var _floatView = new Float32Array(1);
    var _int32View = new Int32Array(_floatView.buffer);

    function floatToHalf(float) {
        _floatView[0] = float;
        var f = _int32View[0];

        var sign = (f >> 31) & 0x0001;
        var exp = (f >> 23) & 0x00ff;
        var frac = f & 0x007fffff;

        var newExp;
        if (exp == 0) {
            newExp = 0;
        } else if (exp < 113) {
            newExp = 0;
            frac |= 0x00800000;
            frac = frac >> (113 - exp);
            if (frac & 0x01000000) {
                newExp = 1;
                frac = 0;
            }
        } else if (exp < 142) {
            newExp = exp - 112;
        } else {
            newExp = 31;
            frac = 0;
        }

        return (sign << 15) | (newExp << 10) | (frac >> 13);
    }

    function packHalf2x16(x, y) {
        return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
    }

    function generateTexture() {
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);

        var texwidth = 1024 * 2; // Set to your desired width
        var texheight = Math.ceil((2 * vertexCount) / texwidth); // Set to your desired height
        var texdata = new Uint32Array(texwidth * texheight * 4); // 4 components per pixel (RGBA)
        var texdata_c = new Uint8Array(texdata.buffer);
        var texdata_f = new Float32Array(texdata.buffer);

        // Here we convert from a .splat file buffer into a texture
        // With a little bit more foresight perhaps this texture file
        // should have been the native format as it'd be very easy to
        // load it into webgl.
        for (let i = 0; i < vertexCount; i++) {
            // x, y, z
            texdata_f[8 * i + 0] = f_buffer[13 * i + 0];
            texdata_f[8 * i + 1] = f_buffer[13 * i + 1];
            texdata_f[8 * i + 2] = f_buffer[13 * i + 2];

            // r, g, b, a
            texdata_c[4 * (8 * i + 7) + 0] = f_buffer[13 * i + 3 + 0];
            texdata_c[4 * (8 * i + 7) + 1] = f_buffer[13 * i + 3 + 1];
            texdata_c[4 * (8 * i + 7) + 2] = f_buffer[13 * i + 3 + 2];
            texdata_c[4 * (8 * i + 7) + 3] = f_buffer[13 * i + 3 + 3];

            // variance
            let sigma = [
                f_buffer[13 * i + 7 + 0],
                f_buffer[13 * i + 7 + 1],
                f_buffer[13 * i + 7 + 2],
                f_buffer[13 * i + 7 + 3],
                f_buffer[13 * i + 7 + 4],
                f_buffer[13 * i + 7 + 5],
            ];

            texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
            texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
            texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
        }

        self.postMessage({ texdata, texwidth, texheight }, [texdata.buffer]);
    }

    function runSort(viewProj) {
        
        if (!buffer) return;
        const f_buffer = new Float32Array(buffer);

        generateTexture();
        // console.time("sort");

        let maxDepth = -Infinity;
        let minDepth = Infinity;
        let sizeList = new Int32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++) {
            let depth =
                ((viewProj[2] * f_buffer[13 * i + 0] +
                    viewProj[6] * f_buffer[13 * i + 1] +
                    viewProj[10] * f_buffer[13 * i + 2]) *
                    1_000_000) |
                0;
            sizeList[i] = depth;
            if (depth > maxDepth) maxDepth = depth;
            if (depth < minDepth) minDepth = depth;
        }
        const totalRange = 512 * 512

        // This is a single-pass counting sort
        let depthInv = totalRange / (maxDepth - minDepth);
        let counts0 = new Uint32Array(totalRange);
        for (let i = 0; i < vertexCount; i++) {
            sizeList[i] = ((sizeList[i] - minDepth) * depthInv) | 0;
            counts0[sizeList[i]]++;
        }
        let starts0 = new Uint32Array(totalRange);
        for (let i = 1; i < totalRange; i++)
            starts0[i] = starts0[i - 1] + counts0[i - 1];
        depthIndex = new Uint32Array(vertexCount);
        for (let i = 0; i < vertexCount; i++)
            depthIndex[starts0[sizeList[i]]++] = i;

        // console.timeEnd("sort");

        lastProj = viewProj;
        self.postMessage({ depthIndex, viewProj, vertexCount }, [
            depthIndex.buffer,
        ]);
    }

    const throttledSort = () => {
        if (!sortRunning) {
            sortRunning = true;
            let lastView = viewProj;
            runSort(lastView);
            setTimeout(() => {
                sortRunning = false;
                if (lastView !== viewProj) {
                    throttledSort();
                }
            }, 0);
        }
    };

    let sortRunning;
    self.onmessage = (e) => {
        if (e.data.buffer) {
            buffer = e.data.buffer;
            vertexCount = e.data.vertexCount;
        } else if (e.data.vertexCount) {
            vertexCount = e.data.vertexCount;
        } else if (e.data.view) {
            viewProj = e.data.view;
            throttledSort();
        }
    };
}

const vertexShaderSource = `
#version 300 es
precision highp float;
precision highp int;

uniform highp usampler2D u_texture;
uniform mat4 projection, view;
uniform vec2 focal;
uniform vec2 viewport;

in vec2 position;
in int index;

out vec4 vColor;
out vec2 vPosition;

void main () {
    uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
    vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1);
    vec4 pos2d = projection * cam;

    float clip = 1.2 * pos2d.w;
    if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
    vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
    mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

    mat3 J = mat3(
        focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z), 
        0., -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z), 
        0., 0., 0.
    );

    mat3 T = transpose(mat3(view)) * J;
    mat3 cov2d = transpose(T) * Vrk * T;

    float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
    float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
    float lambda1 = mid + radius, lambda2 = mid - radius;

    if(lambda2 < 0.0) return;
    vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
    vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
    vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

    vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
    vPosition = position;

    vec2 vCenter = vec2(pos2d) / pos2d.w;
    gl_Position = vec4(
        vCenter 
        + position.x * majorAxis / viewport 
        + position.y * minorAxis / viewport, 0.0, 1.0);

}
`.trim();

const fragmentShaderSource = `
#version 300 es
precision highp float;

in vec4 vColor;
in vec2 vPosition;

out vec4 fragColor;

void main () {
    float A = -dot(vPosition, vPosition);
    if (A < -4.0) discard;
    float B = exp(A) * vColor.a;
    fragColor = vec4(B * vColor.rgb, B);
}

`.trim();


let viewMatrix = getViewMatrix(camera);
async function main() {

    const worker = new Worker(
        URL.createObjectURL(
            new Blob(["(", createWorker.toString(), ")(self)"], {
                type: "application/javascript",
            }),
        ),
    );

    const canvas = document.getElementById("canvas");
    const fps = document.getElementById("fps");

    let projectionMatrix;

    const gl = canvas.getContext("webgl2", {
        antialias: false,
    });

    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, vertexShaderSource);
    gl.compileShader(vertexShader);
    if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(vertexShader));

    const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragmentShader, fragmentShaderSource);
    gl.compileShader(fragmentShader);
    if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS))
        console.error(gl.getShaderInfoLog(fragmentShader));

    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    gl.useProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS))
        console.error(gl.getProgramInfoLog(program));

    gl.disable(gl.DEPTH_TEST); // Disable depth testing

    // Enable blending
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
        gl.ONE_MINUS_DST_ALPHA,
        gl.ONE,
    );
    gl.blendEquationSeparate(gl.FUNC_ADD, gl.FUNC_ADD);

    const u_projection = gl.getUniformLocation(program, "projection");
    const u_viewport = gl.getUniformLocation(program, "viewport");
    const u_focal = gl.getUniformLocation(program, "focal");
    const u_view = gl.getUniformLocation(program, "view");

    // positions
    const triangleVertices = new Float32Array([-2, -2, 2, -2, 2, 2, -2, 2]);
    const vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, triangleVertices, gl.STATIC_DRAW);
    const a_position = gl.getAttribLocation(program, "position");
    gl.enableVertexAttribArray(a_position);
    gl.bindBuffer(gl.ARRAY_BUFFER, vertexBuffer);
    gl.vertexAttribPointer(a_position, 2, gl.FLOAT, false, 0, 0);

    var texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);

    var u_textureLocation = gl.getUniformLocation(program, "u_texture");
    gl.uniform1i(u_textureLocation, 0);

    const indexBuffer = gl.createBuffer();
    const a_index = gl.getAttribLocation(program, "index");
    gl.enableVertexAttribArray(a_index);
    gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
    gl.vertexAttribIPointer(a_index, 1, gl.INT, false, 0, 0);
    gl.vertexAttribDivisor(a_index, 1);

    gl.uniform2fv(u_focal, new Float32Array([camera.fx, camera.fy]));

    projectionMatrix = getProjectionMatrix(
        camera.fx,
        camera.fy,
        innerWidth,
        innerHeight,
    );

    gl.uniform2fv(u_viewport, new Float32Array([innerWidth, innerHeight]));

    gl.canvas.width = innerWidth;
    gl.canvas.height = innerHeight;
    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

    gl.uniformMatrix4fv(u_projection, false, projectionMatrix);

    worker.onmessage = (e) => {
        if (e.data.texdata) {
            const { texdata, texwidth, texheight } = e.data;
            // console.log(texdata)
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_S,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(
                gl.TEXTURE_2D,
                gl.TEXTURE_WRAP_T,
                gl.CLAMP_TO_EDGE,
            );
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RGBA32UI,
                texwidth,
                texheight,
                0,
                gl.RGBA_INTEGER,
                gl.UNSIGNED_INT,
                texdata,
            );
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, texture);
        } else if (e.data.depthIndex) {
            const { depthIndex, viewProj } = e.data;
            gl.bindBuffer(gl.ARRAY_BUFFER, indexBuffer);
            gl.bufferData(gl.ARRAY_BUFFER, depthIndex, gl.DYNAMIC_DRAW);
            vertexCount = e.data.vertexCount;
        }
    };

    const model = await tf.loadGraphModel('http://192.168.1.4:8000/outtfjs4/model.json');
    model.predict(tf.zeros([418]));

    const rowLength = 3 + 4 + 6;
    const apiUrl = "http://172.17.12.143:8001/predict";
    // const apiUrl = "http://10.10.22.246:5000/forward";

    let taskQueue = [];    // 用于存储待处理的消息事件
    let isProcessing = false; // 标识是否正在处理任务
    // console.log(`Using backend: ${tf.getBackend()}`);
    fps.innerText = tf.getBackend()
    await tf.setBackend('webgl');
    await tf.ready();

    // Play audio using AudioContext
    let audioContext = null;

    let vertexCount = 0;
    let play_end = false;
    const frame_in_sec = 1.0/25;
    let count = 0;
    let coeffArray = null;
    let startAudioTime = null;

    // Animation frame rendering function
    function renderFrame(timestamp) {
        if(!play_end ) {
            const audioElapsed = audioContext ? (performance.now() - startAudioTime) / 1000 : 0;
            const expectedFrame = Math.floor(audioElapsed / frame_in_sec);

            if (expectedFrame > count) {
                count = expectedFrame;
                // console.log(`Rendering frame ${count} at audio time ${audioElapsed.toFixed(2)}s`);

                if(count<coeffArray.shape[0]) {
                    tf.engine().startScope();
                    // console.error(count, coeffArray.shape);
                    // console.log(coeffArray.slice([count, 0], [1, -1]).shape);
                    const output = model.predict(coeffArray.slice([count, 0], [1, -1]).reshape([-1]));
    
                    splatData = new Float32Array(output.dataSync());
                    worker.postMessage({
                        buffer: splatData.buffer,
                        vertexCount: Math.floor(splatData.length / rowLength),
                    });
                    tf.engine().endScope();
                    // console.log(Math.floor(splatData.length / rowLength));

                    const viewProj = multiply4(projectionMatrix, viewMatrix);
                    worker.postMessage({ view: viewProj });
            
                    if (vertexCount > 0) {
                        gl.uniformMatrix4fv(u_view, false, viewMatrix);
                        gl.clear(gl.COLOR_BUFFER_BIT);
                        gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, vertexCount);
                    } else {
                        gl.clear(gl.COLOR_BUFFER_BIT);
                    }
                }
            }
        }
        requestAnimationFrame(renderFrame);
    }

    async function sendHttpRequest(request_dict) {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request_dict)
        });
        return response.json();
    }

    async function processQueue() {
        if (isProcessing || taskQueue.length === 0) {
            return;
        }
    
        isProcessing = true;
    
        const event = taskQueue.shift(); // 取出队列中的第一个任务
        try {
            lastFrame = Date.now();
            const result = await sendHttpRequest({"text": event.text, "spkId": event.spkId});

            // Handle audio data
            const audioData = atob(result.audio_data);  // Decode base64 to binary string
            const audioBuffer = new Uint8Array(audioData.length);
            for (let i = 0; i < audioData.length; ++i) {
                audioBuffer[i] = audioData.charCodeAt(i);
            }

            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            audioContext.decodeAudioData(audioBuffer.buffer, buffer => {
                console.log('Audio duration in seconds:', buffer.duration);

                coeffArray = tf.tensor(new Float32Array(result.infer_output.flat()));
                coeffArray = coeffArray.reshape([-1, 103]);
                coeffArray = tf.concat([tf.zeros([coeffArray.shape[0], 300]), 
                                        coeffArray.slice([0, 0], [-1, 100]), 
                                        tf.zeros([coeffArray.shape[0], 3]), 
                                        tf.zeros([coeffArray.shape[0], 3]), 
                                        coeffArray.slice([0, 100], [-1, -1]), 
                                        tf.zeros([coeffArray.shape[0], 6]), 
                                        tf.zeros([coeffArray.shape[0], 3])], -1)
                // console.log('Shape of tensor:', coeffArray.shape);

                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start(0);

                startAudioTime = performance.now();
                count = 0;
                play_end = false;

                source.onended = () => {
                    audioContext.close().then(() => {
                        play_end = true;
                        console.log('AudioContext closed.');
                        audioContext = null;
                    });
                };
                requestAnimationFrame(renderFrame);
            }, error => {
                console.error('Error decoding audio data:', error);
            });

        } catch (error) {
            console.error("Error processing message:", error);
        } finally {
            isProcessing = false;       
            if(taskQueue.length > 0) {   // 如果队列中还有任务，继续处理
                processQueue();
            }
        }
    }

    document.getElementById('processButton').addEventListener('click', () => {
        const textInput = document.getElementById('textInput').value;
        const spkId = document.getElementById('dropdownMenu').value;
        if (textInput.trim()) {
            play_end = false;
            count = 0;
            taskQueue.push({ text: textInput, spkId: spkId });
            processQueue();
        }
    });
}

main().catch((err) => {
    document.getElementById("message").innerText = err.toString();
});
