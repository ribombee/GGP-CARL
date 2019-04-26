import subprocess, sys, paramiko, time, warnings
warnings.filterwarnings(action='ignore', module ='paramiko') 

class Client:
	def __init__(self, ip, password = None):
		if password is None:
			password = raw_input("Enter ssh password")
		self.ip = ip
		self.pw = password
		self.ssh = paramiko.SSHClient()
		self.ssh.load_system_host_keys()
		self.ssh.connect(hostname=self.ip, password=self.pw)
		self.channel = self.ssh.invoke_shell()
		self.buff = self.channel.recv(1024)

	def shell_receive(self):
		while not self.channel.recv_ready():
			time.sleep(0.1)
		self.buff = self.channel.recv(1024)
		return self.buff

	def shell_send(self, message, new_line = True):
		self.channel.send(message + "\n")

	def command(self, message):
		stdin, stdout, stderr = self.ssh.exec_command(message)
		return stdout.read()

class PlayerClient(Client):
	def __init__(self, ip, password = None):
		Client.__init__(self, ip, password)

	def start_player(self, player_type):
		self.shell_send('cd ~/Documents/CarlAgent/GGP-CARL')
		self.shell_send('python play.py ' + player_type + ' 1337')

	def stop_player(self):
		self.shell_send(chr(3), False)

if __name__ == "__main__":
	client = PlayerClient(sys.argv[1], sys.argv[2])
	client.start_player(sys.argv[3])
	time.sleep(10)
	client.stop_player()
	time.sleep(10)
	print client.command("cat .bashrc")
