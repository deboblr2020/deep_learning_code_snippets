
# Creating ssh for Windows and share with WSL

Configuring git with ssh key
- Open git-bash command prompt  
> ssh-keygen
(/c/Users/<username>/.ssh/id_rsa):
- Two keys will get generated - Id_rsa - Id_rsa.pu  
> cat /c/Users/<username>/.ssh/id_rsa.pub
- Copy the content for the id_rsa.pub and paste in the portal under setting information for new key.  

### Sharing the same ssh key with WSL
- On the wsl terminal
> cp -r /mnt/c/Users/<username>/.ssh ~/.ssh

** Nailed it **



