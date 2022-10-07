import torch
import torch.nn.functional as F
import copy

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image by adjusting each pixel of the input image
    perturbation = epsilon*sign_data_grad
    perturbed_image = image + perturbation

    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # Return the perturbed image and perturbation
    return perturbed_image, perturbation

def get_fgsm_samples( model_ref, device, data, targets, epsilon ):

    model = copy.deepcopy(model_ref)

    # Loop over all examples in test set

    # Send the data and label to the device
    data, targets = data.to(device), targets.to(device)

    # Set requires_grad attribute of tensor. Important for Attack
    data.requires_grad = True

    # Forward pass the data through the model
    output = model(data)
    init_preds = output.max(1, keepdim=True)[1] # get the index of the max log-probability

    # TODO for batch testing
    # If the initial prediction is wrong, dont bother attacking, just move on
    # if init_preds.item() != targets.item():
    #     continue

    # Calculate the loss
    loss = F.nll_loss(output, targets)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = data.grad.data

    # Call FGSM Attack
    perturbed_data, perturbations = fgsm_attack(data, epsilon, data_grad)

    return perturbed_data, perturbations

def test_robustness( model, device, test_loader, epsilon ):

    # Accuracy counter
    correct = 0
    adv_examples = []
    adv_perturbations = []

    # Loop over all examples in test set
    for data, targets in test_loader:

        # Send the data and label to the device
        data, targets = data.to(device), targets.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_preds = output.max(1, keepdim=True)[1] # get the index of the max log-probability

        # TODO for batch testing
        # If the initial prediction is wrong, dont bother attacking, just move on
        # if init_preds.item() != targets.item():
        #     continue

        # Calculate the loss
        loss = F.nll_loss(output, targets)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_images, perturbations = fgsm_attack(data, epsilon, data_grad)
        # print("HELLO: ", perturbed_data.shape)
        # break

        # Re-classify the perturbed image
        output = model(perturbed_images)

        # Check for success
        final_preds = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        
        for final_pred, target, perturbed_image, image, p in zip(final_preds, targets, perturbed_images, data, perturbations):

            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                # if (epsilon == 0) and (len(adv_examples) < 100):
                #     adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                #     adv_examples.append( (target.item(), final_pred.item(), adv_ex, data) )
            

            # Save some adv examples if the final prediction is different from target
            if len(adv_examples) < 100:
                adv_examples.append((
                    target.item(),
                    final_pred.item(),
                    perturbed_image,
                    image
                    ))

                # Save perturbations
                adv_perturbations.append( p )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader.dataset))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader.dataset)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples, adv_perturbations