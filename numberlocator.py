import sys, os
from PIL import Image
import tensorflow as tf
from threading              import Thread
from inputs.datasource      import NumberRecognizerDataSource
from models.convolution     import NumberLocator
from image                  import generate_image_patches, draw_bbox

MIN_PATCH_HEIGHT_RELATIVE_TO_IMG = 0.2
MIN_PATCH_HEIGHT_PIXELS = 10
ZOOM_PER_STEP = 0.9
STEP_RELATIVE_TO_PATCH_WIDTH = 0.1
MIN_CLASSIFIER_CONFIDENCE    = 0.99
ASPECT_RATIOS = (1, 1.5, 2, 3, 4)

def get_locator_model():
    model = NumberLocator(input_size       = 64 * 64,
                          fc_layer_sizes    = [1000],
                          fc_keep_prob      = 0.5,
                          filter_sizes      = [5,5,5,1],
                          filter_depths     = [8,16,32,32],
                          pooling_layers    = [1,2,3,4],
                          optimizer         = tf.train.AdamOptimizer(),
                          cutoff            = 0,
                          max_epochs        = 10,
                          L2                = True,
                          test_batch_size   = 500,
                          batch_size        = 500,
                          save_file         = "number_recognizer",
                          name              = "number locator model")
    return model

def evaluate_patches(model, image, patch_size, step_x, step_y, matches):
    for bbox, patch in generate_image_patches(image, patch_size, step_x, step_y):
        label, confidence = model.classify_image(patch)
        if label == 1:
            matches.append((patch, bbox, confidence))
        
def find_number_in_image(model, orig_image, draw_steps=False):
    '''recursive algorithm to locate image patches that contain numbers'''

    def iterate_patches(model, image, patch_size, step_x, step_y, results_accumulator):
        for bbox, patch in generate_image_patches(image, patch_size, step_x, step_y):
            label, confidence = model.classify_image(patch)
            if label == 1:
                results_accumulator.append((patch, bbox, confidence))
            
    def find_best_patch(model, image, cur_bbox=None, numbers_found=None):
        if numbers_found is None: numbers_found = []
        results_accumulator = []
        dim = min(image.size) * ZOOM_PER_STEP
        for i in range(2):
            if dim < max(MIN_PATCH_HEIGHT_PIXELS, orig_image.height * MIN_PATCH_HEIGHT_RELATIVE_TO_IMG):
                continue
            threads = []
            step = dim * STEP_RELATIVE_TO_PATCH_WIDTH
            for aspect_ratio in ASPECT_RATIOS:
                thread = Thread(target=iterate_patches, args=(model, image.copy(), (dim * aspect_ratio, dim), step, step, results_accumulator))
                threads.append(thread)        
            for thread in threads:
                thread.start()
                thread.join()
            dim = int(dim * .75)
            if len(results_accumulator) > 0:
                break
            
        if len(results_accumulator) == 0:
            #base case: no patches left
            return numbers_found
        results_accumulator.sort(key=lambda x: x[2], reverse=True)
        best_patch, best_bbox, best_confidence = results_accumulator[0]
        if not cur_bbox:
            bbox_absolute_coords = best_bbox
        else:
            #calculate bounding box relative to original image
            x_offset, y_offset = cur_bbox[:2]
            bbox_absolute_coords = best_bbox[0] + x_offset, best_bbox[1] + y_offset, best_bbox[2] + x_offset, best_bbox[3] + y_offset
            if draw_steps:
                print best_confidence
                draw_bbox(orig_image, bbox_absolute_coords)
        numbers_found.append((bbox_absolute_coords, best_confidence))
        return find_best_patch(model, best_patch, cur_bbox=bbox_absolute_coords, numbers_found=numbers_found)
    found_numbers = find_best_patch(model, orig_image)
    high_confidence_matches = [bbox for (bbox, confidence) in found_numbers if confidence > MIN_CLASSIFIER_CONFIDENCE]
    #add the entire image to the list as well
    high_confidence_matches.append((0, 0, orig_image.width, orig_image.height))
    return high_confidence_matches
    

if __name__ == "__main__":
    args  = sys.argv[1:]
    model = get_locator_model()
    if args[0] == "train":
        data  = NumberRecognizerDataSource("extra", (1, 200000), "test", (1,13000))
        model.train(data)
    elif args[0] == "find":
        imgfile = args[1]
        image = Image.open(imgfile)
        bboxes = find_number_in_image(model, image)
        for bbox in bboxes:
            draw_bbox(image, bbox)
        image.show()
    
