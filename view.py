from IPython.display import HTML
from base64 import b64encode
mp4 = open('video2.mp4','rb').read()
data_url = "data:video2/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls autoplay loop>
    <source src="%s" type="video2/mp4">
</video>
""" % data_url)

<video width="320" height="240" controls>
  <source src="video2.mp4" type="video/mp4">
#   <source src="movie.ogg" type="video/ogg">
Your browser does not support the video tag.
</video>


print('done...')