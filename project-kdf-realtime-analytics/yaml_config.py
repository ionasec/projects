"""Load configuration from .yaml file."""
import confuse

print(confuse.Filename())

config = confuse.Configuration('project-kdf-realtime-analytics')
#runtime = config['AWS']['Lambda']['Runtime'].get()
#runtime = config['AWS']['Region'].get()
#print(runtime)