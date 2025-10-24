module.exports = {
  apps: [{
    name: 'instantmesh-service',
    script: '/home/kobe/miniconda3/envs/instantmesh/bin/python',
    args: '/home/kobe/404-gen/v1/3D-gen/generation/services/instantmesh_service.py',
    cwd: '/home/kobe/404-gen/v1/3D-gen/generation',
    interpreter: 'none',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    max_memory_restart: '8G',
    env: {
      PYTHONPATH: '/tmp/InstantMesh',
      CUDA_VISIBLE_DEVICES: '0',  // GPU 0 (RTX 4090) - shared with FLUX via lazy loading
      PATH: '/home/kobe/miniconda3/envs/instantmesh/bin:' + process.env.PATH,
      TORCH_CUDA_ARCH_LIST: '8.9',
      CC: '/home/kobe/miniconda3/envs/instantmesh/bin/gcc',
      CXX: '/home/kobe/miniconda3/envs/instantmesh/bin/g++'
    },
    error_file: '/home/kobe/.pm2/logs/instantmesh-service-error.log',
    out_file: '/home/kobe/.pm2/logs/instantmesh-service-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z'
  }]
};
