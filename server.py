import web, os
from cStringIO import StringIO
from PIL import Image
from web import form
from readnumber       import get_best_svhn_model, locate_and_read_number
from image            import extract_data_from_image
from inputs.svhn      import process_image_by_file
from numberlocator    import get_locator_model
from flags            import FLAGS

"""
simple web application for reading a number in an uploaded image
"""

urls = ('/',            'Upload',
        '/images/(.*)', 'Images')

render = web.template.render('templates')
myform = form.Form(form.File("image"), form.Button("submit"))

number_locator = get_locator_model()
number_reader  = get_best_svhn_model()
  
if not os.path.exists(FLAGS.uploaded_img_dir):
  os.mkdir(FLAGS.uploaded_img_dir)

class Index(object):

  def GET(self):
    raise web.seeother("/upload")
    
class Upload(object):
  
  def GET(self):
    return render.classifier(myform())
  

  def POST(self):
    form = web.input(image={}, classifiers="")
    imgfile = form['image']
    path = os.path.join(FLAGS.uploaded_img_dir, imgfile.filename)
    f = StringIO()
    if not imgfile.value:
      raise web.seeother("/upload")
    f.write(imgfile.value)
    image = Image.open(f)
    prediction = locate_and_read_number(image, number_locator, number_reader)
    if not prediction:
      return render.result(False, None, None)
    image.save(path)
    return render.result(True, imgfile.filename, prediction)


class Images(object):

  def GET(self, name):
    print name
    ext = name.split(".")[-1]
    cType = {
            "png"  : "images/png",
            "jpg"  : "images/jpeg",
            "jpeg" : "images/jpeg",
            "gif"  : "images/gif",
            }
    if name in os.listdir(FLAGS.uploaded_img_dir):
      web.header("Content-Type", cType[ext])
      return open(os.path.join(FLAGS.uploaded_img_dir, name), "rb").read()
    else:
      raise web.notfound()

application = web.application(urls, globals()).wsgifunc()

if __name__ == "__main__":
  from werkzeug.serving import run_simple
  run_simple('0.0.0.0', 8080, application)
