module.exports = {
  apps : [{
    name: 'miner-competitive',
    script: 'serve_miner_competitive.py',
    interpreter: '/root/miniconda3/envs/three-gen-neurons/bin/python',
    args: '--wallet.name validator --wallet.hotkey sn17miner2 --subtensor.network finney --netuid 17 --generation.endpoint http://127.0.0.1:8093 --logging.debug'
  }]
};
