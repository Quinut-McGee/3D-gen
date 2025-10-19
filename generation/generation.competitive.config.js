module.exports = {
  apps : [{
    name: 'generation-competitive',
    script: 'serve_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-mining/bin/python',
    args: '--port 8093 --config configs/text_mv_fast.yaml --flux-steps 4 --validation-threshold 0.6 --enable-validation'
  }]
};
