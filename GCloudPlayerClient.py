import subprocess, sys, paramiko, time, warnings, os
warnings.filterwarnings(action='ignore', module ='paramiko') 

class Client:
	def __init__(self, ip):
		self.ip = ip
		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.load_system_host_keys()
		self.ssh.connect(hostname=self.ip, timeout=10, auth_timeout=10, banner_timeout=10)
		self.channel = self.ssh.invoke_shell()
		self.buff = self.channel.recv(1024)

	def shell_receive(self, buffer_size = 1024):
		while not self.channel.recv_ready():
			time.sleep(0.1)
		self.buff = self.channel.recv(buffer_size)
		return self.buff

	def shell_send(self, message, new_line = True):
		command = message
		if new_line:
			command += "\n"
		time.sleep(0.2)
		self.channel.send(command)

	def command(self, message, as_sudo = True):
		command = message
		if as_sudo:
			if "~" in command:
				command = command.replace("~", "/root")
			command = "sudo sh -c \"" + message + "\""
		_, stdout, _ = self.ssh.exec_command(command)
		return stdout.read()

class PlayerClient(Client):
	def __init__(self, ip):
		Client.__init__(self, ip)

	def enter_project_dir(self):
		self.shell_send('sudo su')
		self.shell_send('cd ~/Documents/CarlAgent/GGP-CARL')

	def update_player_repo(self):
		self.shell_send("git pull")
        time.sleep(3)

	def start_player(self, player_type, regressor_type = "sgd", port = 1337, max_expansions = 100000):
		self.shell_send('python play.py ' + player_type + ' ' + str(port) + ' ' + regressor_type + ' ' + str(max_expansions)) 

	def stop_player(self):
		self.shell_send(chr(3), False)

if __name__ == "__main__":
	client = PlayerClient(sys.argv[1])
	client.enter_project_dir()
	client.update_player_repo()
	client.shell_send("ls")
	time.sleep(10)
	print client.shell_receive()
	#print client.command("cd ~; ls")
