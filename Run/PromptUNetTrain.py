from train_neat import *
from PromptUNet.PromptUNet import PromptUNet
from utils import *
from test_on_diff_data import plot_output

def prompt_train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"attention training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")
def prompt_train_batch(images,labels,points,point_labels,model,optimizer,criterion,plot=False):
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    point_input,point_label_input = torch.from_numpy(points).to(device,dtype=torch.float),torch.from_numpy(point_labels).to(device,dtype=torch.float)
    outputs = model(images,point_input,point_label_input,train_attention=True)
    loss = criterion(outputs, labels)
    if plot:
        point_tuples = [(i,j,val[0]) for (i,j),val in zip(points[0,:,:],point_labels[0,:,:])]
        print(f'point_tuples: {point_tuples}')
        plot_output(outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1),images,labels,loss,point_tuples,detach=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    new_points,new_point_labels = generate_points_batch(labels,outputs,detach=True)

    point_labels = np.concatenate([point_labels,new_point_labels],axis=1)
    points = np.concatenate([points,new_points],axis=1)
    return loss,points,point_labels
def encoder_decoder_train_batch(images,labels,model,optimizer,criterion,plot=False):
    images,labels = images.to(device,dtype=torch.float32),labels.to(device,dtype=torch.float32)
    outputs = model(images, [], [], train_attention=False)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

def initial_test(images,labels,model,criterion,):
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    outputs = model(images, [], [], train_attention=False)
    points, point_labels = generate_points_batch(labels, outputs)
    point_tuples = [(i, j, val[0]) for (i, j), val in zip(points[0, :, :], point_labels[0, :, :])]

    outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    score = criterion(outputs,labels)
    plot_output(outputs,images,labels,score[1],point_tuples)
def prompt_test(model,test_loader,criterion,config,best_valid_score,example_ct,num_points=5,plot=False):
    model.eval()

    with torch.no_grad():

        val_scores = np.zeros(5)
        f1_score_record = np.zeros((4,5))
        total = 0
        for _,(images,labels) in enumerate(test_loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            points, point_labels = generate_points_batch(labels, torch.from_numpy(np.zeros((config.batch_size, 1, 512, 512), np.uint8)).to(device))

            for i in range(num_points):

                point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float), torch.from_numpy(point_labels).to(device, dtype=torch.float)
                outputs = model(images, point_input, point_label_input)

                new_points, new_point_labels = generate_points_batch(labels, outputs)

                outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

                score = criterion(outputs,labels)


                val_scores[i] += score[1].item() / 2 + score[2].item() / 2

                f1_score_record[:,i] += score
                if plot:
                    point_tuples = [(i, j, val[0]) for (i, j), val in zip(points[0, :, :], point_labels[0, :, :])]
                    print(f'point_tuples: {point_tuples}')
                    plot_output(outputs,images, labels, val_scores[i], point_tuples,detach=False)

                point_labels = np.concatenate([point_labels, new_point_labels], axis=1)
                points = np.concatenate([points, new_points], axis=1)

            total += labels.size(0)

    f1_score_record /= total
    val_scores /= total
    val_score_str = ', '.join([format(score,'.8f') for score in val_scores])
    disc_scores = ', '.join([format(score,'.8f') for score in f1_score_record[3,:]])
    cup_scores = ', '.join([format(score,'.8f') for score in f1_score_record[2, :]])

    return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}"


    data_to_log = {}




    # Loop through the validation scores and add them to the dictionary
    for i, val_score in enumerate(val_scores):
        data_to_log[f"val_score {i + 1} points"] = val_score
        data_to_log[f"Validation Background F1 Score {i + 1}"] = f1_score_record[0][i]
        data_to_log[f"Validation Disc F1 Score {i + 1}"] = f1_score_record[3][i]
        data_to_log[f"Validation Cup F1 Score {i + 1}"] = f1_score_record[2][i]
        data_to_log[f"Validation Outer Ring F1 Score {i + 1}"] = f1_score_record[1][i]

    wandb.log(data_to_log,step=example_ct)
    model.train()

    if val_scores[-1] > best_valid_score[0]:
        data_str = f"Valid score for point {len(val_scores)} improved from {best_valid_score[0]:2.8f} to {val_scores[-1]:2.8f}. Saving checkpoint: {config.low_loss_path}"
        print(data_str)
        best_valid_score[0] = val_scores[-1]
        torch.save(model.state_dict(), config.low_loss_path)
        save_model(config.low_loss_path, "low_loss_model")

    return return_str


def train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"attention training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")
def prompt_train(model, loader,test_loader, criterion, eval_criterion, config,num_points=5):

    wandb.watch(model,criterion,log='all',log_freq=50) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0
    optimizer = torch.optim.Adam(model.parameters(), lr)
    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()
        counter = 0
        for _,(images, labels) in enumerate(loader):
                points, point_labels = generate_points_batch(labels.to(device), torch.from_numpy(np.zeros((config.batch_size, 1, 512, 512), np.uint8)).to(device))
                if counter%7==0:
                    plot = True
                else:
                    plot = False
                counter+=1
                for i in range(num_points):
                    loss,points,point_labels = prompt_train_batch(images,labels,points,point_labels,model,optimizer,criterion,plot)
                avg_epoch_loss += loss
                example_ct += len(images)
                batch_ct +=1

                if ((batch_ct+1)%4)==0:
                    prompt_train_log(loss,batch_ct,epoch)




        test_results = prompt_test(model,test_loader,eval_criterion,config,best_valid_score,batch_ct)
        avg_epoch_loss/=len(loader)
        end_time = time.time()
        iteration_mins,iteration_secs = train_time(start_time,end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += test_results
        print(data_str)
    torch.save(model.state_dict(),config.final_path)
    save_model(config.final_path,"final_model")


def prompt_train_from_prev_model(model, loader, test_loader, criterion, eval_criterion, config, num_points=5):
    unet_path = "/home/kebl6872/Desktop/new_data/REFUGE/test/unet_batch_lr_0.0003_bs_16_fs_12_[6_12_24_48]/Checkpoint/seed/279/lr_0.0003_bs_16_lowloss.pth"
    check_point = torch.load(unet_path)
    model.load_state_dict(check_point,strict=False)
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze promptImageCrossAttention and promptSelfAttention
    for param in model.promptImageCrossAttention.parameters():
        param.requires_grad = True
    for param in model.promptSelfAttention.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    wandb.watch(model, criterion, log='all', log_freq=50)  # this is freq of gradient recordings
    example_ct = 0
    batch_ct = 0

    #check initialised weights provide good baseline performance
    # for i, (images,labels) in enumerate(test_loader):
    #     if i > 5:
    #         break
    #     with torch.no_grad():
    #         initial_test(images,labels,model,eval_criterion)


    best_valid_score = [0.0]  # in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()
        counter = 0
        for _, (images, labels) in enumerate(loader):

            points, point_labels = generate_points_batch(labels.to(device), torch.from_numpy(
                np.zeros((config.batch_size, 1, 512, 512), np.uint8)).to(device))

            for i in range(num_points):
                if counter % 7 == 0:
                    plot = True
                else:
                    plot = False
                counter += 1
                loss, points, point_labels = prompt_train_batch(images, labels, points, point_labels, model, optimizer,
                                                                criterion, plot)
                print(f'point {i}: {loss}')
            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct += 1

            if ((batch_ct + 1) % 4) == 0:
                prompt_train_log(loss, batch_ct, epoch)

        test_results = prompt_test(model, test_loader, eval_criterion, config, best_valid_score, batch_ct)
        avg_epoch_loss /= len(loader)
        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += test_results
        print(data_str)
    torch.save(model.state_dict(), config.final_path)
    save_model(config.final_path, "final_model")


def prompt_make(config):
    if config.dataset=="GS1":
        gs1 = True
    else:
        gs1 = False

    if config.dataset == "GAMMA":
        gamma=True
    else:
        gamma=False
    train,test = get_data(train=True,transform=config.transform,gs1=gs1,gamma=gamma),get_data(train=False,gs1=gs1,gamma=gamma)
    eval_criterion = f1_valid_score
    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)
    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5,lambda_ce=0.5)

    model = PromptUNet(config.device,3,config.classes,config.base_c,)


    return model,train_loader,test_loader,criterion,eval_criterion
def prompt_model_pipeline(hyperparameters):
    with wandb.init(project="junk",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = prompt_make(config)
        # print(model)
        model = model.to(device)
        prompt_train_from_prev_model(model,train_loader,test_loader,criterion,eval_criterion,config)

    return model


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)
    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')


    for _ in range(no_runs):
        config = dict(epochs=1000, classes=3, base_c = 12, kernels=[6,12,24,48], norm_name=norm_name,
                      batch_size=batch_size, learning_rate=lr, dataset="GAMMA",
                      architecture=model_name,seed=401,transform=True,device=device)
        config["seed"] = randint(601,800)
        seeding(config["seed"])


        data_save_path = f'/home/kebl6872/Desktop/new_data/{config["dataset"]}/test/prompt{model_name}_{norm_name}_lr_{lr}_bs_{batch_size}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/'
        create_dir(data_save_path + f'Checkpoint/seed/{config["seed"]}')
        checkpoint_path_lowloss = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
        checkpoint_path_final = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
        create_file(checkpoint_path_lowloss)
        create_file(checkpoint_path_final)
        config['low_loss_path']=checkpoint_path_lowloss
        config['final_path'] = checkpoint_path_final

        model = prompt_model_pipeline(config)


