###### calculate factor for inter_sample
output_inter = output.clone().detach() / T_inter  # same as data.clone(), the purpose is to detach to prevent backpropagating
output_inter_gt = torch.softmax(output_inter, dim=1) * target_one_hot
output_inter_factor = 1.0 - torch.sum(output_inter_gt, dim=1, keepdim=True)
###### calculate factor for intra_sample
output_intra = output.clone().detach() / T_intra  # same as data.clone(), the purpose is to detach to prevent backpropagating
output_intra_gt = torch.softmax(output_intra, dim=1) * target_one_hot
output_intra_factor = 1.0 - torch.sum(output_intra_gt, dim=1, keepdim=True)

grad_design = (output_inter_factor / output_intra_factor) * (weight_probability * torch.softmax(output_intra, dim=1) - target_one_hot)
loss = torch.sum(grad_design * output) / output.size(0)
