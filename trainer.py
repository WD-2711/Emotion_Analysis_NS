#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
# @Time  : 2023/03/21 19:39:49
# @Author: wd-2711
'''

from config import *
from model.bert import *
from model.bert_textcnn import *
from dataset import *

def setup_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True

def trainer(m, d:list, args:dict, fold_i:int) -> tuple:
  model_name = args["model_name"]
  device = args["device"]
  batch_size, num_epochs = args["batch_size"], args["epoches"]
  lr_rate, weight_decay = args["learning_rate"], args["weight_decay"]


  def log_rmse(flag:bool, net, train_d:EmoDataset) -> tuple:
    net.eval()
  
    corr = 0
    loss_total = 0
    for i, b in enumerate(train_d):
        b = [p.to(device) for p in b]
        out = net([b[0], b[1], b[2]])
        loss = loss_fn(out, b[3])
        out = out.data.max(dim=1, keepdim=True)[1]
        out = torch.squeeze(out)
        corr += torch.sum(torch.eq(out, b[3])).item()
        loss_total += loss.item()
      
    # accuracy = corr*100.0/(len(train_d)*batch_size)
    accuracy = corr*100.0/len(train_d.dataset)
    loss = loss_total/len(train_d)
    net.train()
    return (loss, accuracy)
  
  train_ls, valid_ls = [], []
  train_d, valid_d = d[0], d[1]
  train_d, valid_d = Data.DataLoader(
      dataset=EmoDataset(train_d, args), 
      batch_size=batch_size, 
      shuffle=True, 
      num_workers=1
  ), Data.DataLoader(
      dataset=EmoDataset(valid_d, args), 
      batch_size=batch_size, 
      shuffle=True, 
      num_workers=1
  )
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(m.parameters(), lr=lr_rate, weight_decay=weight_decay)

  for e in range(num_epochs):
    st = time.time()
    for i, batch in enumerate(train_d):
      optimizer.zero_grad()
      batch = tuple(p.to(device) for p in batch)
      pred = m([batch[0], batch[1], batch[2]])
      loss = loss_fn(pred, batch[3])
      loss.backward()
      optimizer.step()

    train_ls.append(log_rmse(0, m, train_d)) 
    valid_ls.append(log_rmse(1, m, valid_d))
    if valid_ls[-1][0] < args["loss"]:
      model_path = args["model_path"]+"_{}_ls{:.2f}.pt".format(args["model_type"], valid_ls[-1][0])
      torch.save(m.state_dict(), model_path)
    print("[-] fold:{:3d} epoch:{:3d} | train loss:{:7.4f} accr:{:3.2f} | valid loss:{:7.4f} accr:{:3.2f} | time cost {:7.1f}s".format(
        fold_i, e, train_ls[-1][0], train_ls[-1][1], valid_ls[-1][0], valid_ls[-1][1], time.time()-st
    ))

  return (train_ls, valid_ls)
  
def get_k_fold_data(k:int, i:int, data:list) -> list:
  assert len(data[0]) == len(data[1]), "length error"
  fold_size = len(data[0]) // k

  train_data, valid_data = None, None
  for j in range(k):
    slice_data = [
        data[0][j*fold_size:(j+1)*fold_size],
        data[1][j*fold_size:(j+1)*fold_size],
    ]
    if j == i:
      valid_data = slice_data
    elif train_data == None:
      train_data = slice_data
    else:
      train_data[0].extend(slice_data[0]), train_data[1].extend(slice_data[1])
  return [train_data, valid_data]

def k_fold(args:dict):
  k, model_name, divice = args["fold_k"], args["model_name"], args["device"]
  train_loss_sum, valid_loss_sum = 0, 0
  train_acc_sum, valid_acc_sum = 0, 0
  
  data = data_reader(args)
  setup_seed(20)
  print("[+] Deivce", device)
  print("[+] Model:{}".format(args["model_type"]))
  print("[+] Source data:{}".format(args["data_path"]))
  print("[+] Batch size:{:3d} | lr_rate:{:5.4f} | max_len:{:3d}".format(args["batch_size"], args["learning_rate"], args["max_len"]))
  for i in range(k):
    d = get_k_fold_data(k, i, data)
    if args["model_type"] == "bert":
      m = BertAnalysis(args).to(args["device"])
    elif args["model_type"] == "bert_textcnn":
      m = BertBlendTextCNN(args).to(args["device"])
    (train_ls, valid_ls) = trainer(m, d, args, i)

    train_loss_sum += train_ls[-1][0]
    valid_loss_sum += valid_ls[-1][0]
    train_acc_sum += train_ls[-1][1]
    valid_acc_sum += valid_ls[-1][1]
    torch.cuda.empty_cache()

  print("[+] Result | train loss:{:7.4f} accr:{:3.2f} | valid loss:{:7.4f} accr:{:3.2f}".format(
        train_loss_sum/k, train_acc_sum/k, valid_loss_sum/k, valid_acc_sum/k
  ))

def tester(args:dict):
  def load_and_test(p:str, args:dict):
    lr_rate, weight_decay, batch_size = args["learning_rate"], args["weight_decay"], args["batch_size"]
    test_data = data_reader(args, with_labels=False)

    test_data = Data.DataLoader(dataset=EmoDataset(test_data, args, with_labels=False), batch_size=batch_size, shuffle=False, num_workers=1)
    if "bert_textcnn" in p:
      m = BertBlendTextCNN(args).to(args["device"])
    else:
      m = BertAnalysis(args).to(args["device"])
    m.eval()
    state = torch.load(p)
    m.load_state_dict(state)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(m.parameters(), lr=lr_rate, weight_decay=weight_decay)
    
    result_df = None
    for i, b in enumerate(test_data):
      b = [p.to(device) for p in b]
      out = m([b[0], b[1], b[2]])
      out = out.data.max(dim=1, keepdim=True)[1]
      out = torch.squeeze(out)
      df_slice = pd.DataFrame({
        "index": [ii for ii in range(i*batch_size, (i+1)*batch_size)],
        "prediction": out.cpu().numpy()
      })
      if result_df is None: 
        result_df = df_slice
      else:
        result_df = pd.concat([result_df,df_slice], axis=0)
    result_path = "./result/" + p.split("/")[-1][:-3] + ".tsv"
    result_df.to_csv(result_path, sep="\t", index=False)

  list_dirs = os.walk(args["test_model_dir"])
  for root, _, files in list_dirs:
    for f in tqdm(files):
      path = os.path.join(root, f)
      load_and_test(path, args)



