module.exports = {
  apps : [{
    name: 'miner-competitive',
    script: 'serve_miner_competitive.py',
    interpreter: '/home/kobe/miniconda3/envs/three-gen-mining/bin/python',
    args: '--wallet.name validator --wallet.hotkey sn17miner2 --subtensor.network finney --netuid 17 --logging.debug',
    env: {
      CUDA_VISIBLE_DEVICES: '0'            // Ensure miner also sees only GPU 0
    }
  }]
};
