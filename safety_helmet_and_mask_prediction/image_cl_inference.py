import cv2
from PIL import Image
from train_image_classification import Network as nt
def process_image(image_path):
    # Load Image
    img = plt.imread(image_path)
    img=img[:,:,:3]
    # Get the dimensions of the image
    width, height,c = img.shape
    print(img.shape)
    # Resize by keeping the aspect ratio, but changing the dimension
    # so the shortest size is 255px
    #img = img.resize((255, int(255*(height/width))) if width < height else (int(255*(width/height)), 255))
    #img = img.resize((28, 28))
    # Get the dimensions of the new image size
    #width, height = img.size
    
    # Turn image into numpy array
    img = np.array(img)
    print(img.shape)
    # Make the color channel dimension first instead of last
    img = img.transpose((2, 0, 1))
    
    # Make all values between 0 and 1
    img = img/255
    
    # Normalize based on the preset mean and standard deviation
    img[0] = (img[0] - 0.485)/0.229
    img[1] = (img[1] - 0.456)/0.224
    img[2] = (img[2] - 0.406)/0.225
    
    # Add a fourth dimension to the beginning to indicate batch size
    img = img[np.newaxis,:]
    
    # Turn into a torch tensor
    image = torch.from_numpy(img)
    print(image.size)
    image = image.float()
    if train_on_gpu:
        image = image.cuda()
    return image
def predict(image, model):
    # Pass the image through our model
    output = model.forward(image)
    
    # Reverse the log function in our output
    #output = torch.exp(output)
    
    # Get the top predicted class, and the output percentage for
    # that class
    probs, classes = output.topk(1, dim=1)
    return probs.item(), classes.item()

def test_pridiction(data_dir_test, test_img, network):
    top_class = []
    top_prob = []
    submission = pd.DataFrame(columns=['Name', 'Label'])
    submission['Name'] = test_img
    for i in range(0, len(test_img)):
        image = process_image('{}/{}'.format(data_dir_test, test_img[i]))# Give image to model to predict output
        prob, pred_class = predict(image, network)
        top_prob.append(prob)
        top_class.append(pred_class)
    submission['Label'] = top_class
    return top_class, top_prob, submission

data_dir_test = "yolo_results"
test_img = os.listdir(data_dir_test)
print(test_img)
#network = torch.load(../working/model_detail.pt)
#top_class, top_prob, submission = test_pridiction(data_dir_test, test_img, network)

model = nt.Network()
model.load_state_dict(torch.load('model_detail.pt'))
print(model.eval())

top_class, top_prob, submission = test_pridiction(data_dir_test, test_img, model)

print(top_class, top_prob, submission)
