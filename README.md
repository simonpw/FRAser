# FRAser

usage: fraser [-h] [-u] [-c country] [-f features] [-p protocols] [--list-protocols] [--server-url] [-v]

A command line frequency response analyser, written in Python, Version 1.0

## Options:
  
  `-h, --help` - Show help message and exit  
  `-v, --verbose` - Be verbose  
  `-q, --quiet` - Be quiet  
  `-p, --plots` - Show plots, requires matplotlib
  `-l, --list-devices` - List all available devices and exit  
  `-i INPUT, --input INPUT` - The input device to use, required  
  `-o OUTPUT, --output OUTPUT` - The output device to use, required  
  `-c CHANNELS, --channels CHANNELS` - The number of channels to use, default: 1  
  `-r RATE, --rate RATE` - The sample rate  
  `-t TIME, --time TIME` - The sampling time  
  `--min-freq FREQ` - The minimum FFT frequency, default: 10Hz  
  `--max-freq FREQ` - The maximum FFT frequency, default 22kHz  
  `--fft-window WINDOW` - The FFT window size, default: 8192  
  `--fft-threads THREADS` - The number of threads to use for the FFT calculation, default -1 = cpu_count  
  `-s, --save` - Save the input data to a WAV file
  `--save-filename FILENAME` - Where to save input data, default: `[HOME]/.cache/fraser/FRAser.wav`  
  `--log LEVEL` - Logging level to use, default: WARNING  
  `--logfile` - Where to save the log, default: `[HOME]/.cache/fraser/FRAser.log`  
  
## Countries:  
Countries can be selected using their english name or ISO3166 code, some abbreviations will work but cannot be guaranteed.  
Use `any` to explicitly match any country, overrides the default in configuration file.
  
## Features:
The following selections can be made using the -f argument:  
  
- `any` : Don't filter based on features, overrides the config file default  
- `sta` : Standard VPN servers  
- `ded` : Dedicated IP address servers  
- `doub` : Double VPN servers  
- `obf` : Obfuscated servers  
- `p2p` : Servers optimised for P2P usage  
- `tor` : Onion over VPN servers  
  
For details of these servers and their intended usage check the NordVPN documentation.  

## Protocols:

The following protocols can be selected with the -p argument:  
  
- `ikev2`  
- `openvpn_udp`  
- `openvpn_tcp`  
- `socks`  
- `proxy`  
- `pptp`  
- `l2tp`  
- `openvpn_xor_udp`  
- `openvpn_xor_tcp`  
- `proxy_cybersec`  
- `proxy_ssl`  
- `proxy_ssl_cybersec`  
- `ikev2_v6`  
- `open_udp_v6`  
- `open_tcp_v6`  
- `wireguard_udp`  
- `openvpn_udp_tls_crypt`  
- `openvpn_tcp_tls_crypt `  
- `openvpn_dedicated_udp`  
- `openvpn_dedicated_tcp`  
- `skylark`  
- `mesh_relay `
  
`--list-protcols` will list these options.  
For details of these protocols and their usage consult the NordVPN documentation.  

## Configuration file:

A user configuration file can be used in this location: /home/[USER]/.config/nordquery/nordquery.conf  
  
The structure is:  
  
`[defaults]`  
`always_update = yes/no` - forces an update of the server database with every query  
`country = xx` or `any` - an ISO3166 code for the default country  
`features = xxx xxx` or `any`  - a list of the default features to use, e.g. `sta p2p`  
`db_path` - path to store the server database file, overrides the default (`/home/[USER]/.cache/nordquery`)  
`db_filename` - name for server database file, overrides the default (`server.db`)
  
These settings will be overriden by command line arguments.
  

Simon Williams 14/04/22  
simon@clockcycles.net
