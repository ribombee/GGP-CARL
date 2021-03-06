import subprocess, sys, paramiko, time, warnings
warnings.filterwarnings(action='ignore', module ='paramiko') 

class Client:
	def __init__(self, ip, password = None):
		self.ip = ip
		self.pw = password
		self.ssh = paramiko.SSHClient()
		self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		self.ssh.connect(hostname=self.ip, password=self.pw)
		transport = self.ssh.get_transport()
		transport.set_keepalive(60)
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

	def command(self, message):
		_, stdout, _ = self.ssh.exec_command(message)
		return stdout.read()

class PlayerClient(Client):
	def __init__(self, ip, password = None):
		Client.__init__(self, ip, password)

	def enter_project_dir(self):
		self.shell_send('cd ~/Documents/CarlAgent/GGP-CARL')

	def update_player_repo(self):
		self.shell_send("git pull")
        time.sleep(3)

	def start_player(self, player_type, regressor_type = "sgd", port = 1337, max_expansions = -1):
		self.shell_send('python play.py ' + player_type + ' ' + str(port) + ' ' + regressor_type + ' ' + str(max_expansions)) 

	def stop_player(self):
		self.shell_send(chr(3), False)

if __name__ == "__main__":
	client = PlayerClient(sys.argv[1], sys.argv[2])
	client.start_player(sys.argv[3])
	time.sleep(10)
	client.stop_player()
	time.sleep(10)
	print client.command("cat .bashrc")
