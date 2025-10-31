module.exports = {
  apps: [{
    name: 'trellis-microservice',
    script: '/home/kobe/miniconda3/envs/trellis-env/bin/python',
    args: 'generation/serve_trellis.py --host 0.0.0.0 --port 10008',
    cwd: '/home/kobe/404-gen/v1/3D-gen',
    interpreter: 'none',
    instances: 1,
    exec_mode: 'fork',
    autorestart: true,
    watch: false,
    max_memory_restart: '8G',
    env: {
      CUDA_VISIBLE_DEVICES: '0',
      SPCONV_ALGO: 'native',
      PYTHONPATH: '/home/kobe/404-gen/v1/3D-gen:/home/kobe/404-gen/v1/3D-gen/TRELLIS',
    },
    error_file: '/home/kobe/.pm2/logs/trellis-microservice-error.log',
    out_file: '/home/kobe/.pm2/logs/trellis-microservice-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    min_uptime: '10s',
    max_restarts: 10,
    restart_delay: 5000,
  }]
};
