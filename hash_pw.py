from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()
password = "Gop65792@"  # Replace with your desired password
hashed = bcrypt.generate_password_hash(password).decode()
print("Hashed password:", hashed)
