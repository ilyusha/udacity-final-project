from tensorflow import flags

flags.DEFINE_integer("img_height", 64, "height, in pixels, of input image")
flags.DEFINE_integer("img_width",  64, "width, in pixels, of input image")
flags.DEFINE_integer("synthetic_img_height",  56, "width, in pixels, of synthetic sample image")
flags.DEFINE_integer("synthetic_img_width",   56, "width, in pixels, of synthetic sample image")
flags.DEFINE_string("generated_img_dir", "data/generated", "directory of synthetic images")
flags.DEFINE_string("svhn_data_dir", "data/svhn", "directory of svhn data")
flags.DEFINE_string("svhn_train_dir", "data/svhn/train", "directory of svhn train data")
flags.DEFINE_string("svhn_test_dir", "data/svhn/test", "directory of svhn test data")
flags.DEFINE_string("svhn_extra_dir", "data/svhn/extra", "directory of svhn extra data")
flags.DEFINE_string("classifier_dir", "classifiers", "directory for storing saved classifier data")
flags.DEFINE_string("uploaded_img_dir", "uploaded", "directory for storing uploaded images")

flags.DEFINE_bool("train", False,     "train a classifier")
flags.DEFINE_bool("synthetic", False, "use synthetic dataset")
flags.DEFINE_bool("svhn", False,      "use SVHN dataset")
flags.DEFINE_bool("joint", False,     "train multi-logit model")
flags.DEFINE_integer("digit", False,  "train model for the specified digit position")
flags.DEFINE_bool("length", False,    "train model on the length of the number")
flags.DEFINE_string("run", "",   "read the number anywhere in an image")

FLAGS = flags.FLAGS