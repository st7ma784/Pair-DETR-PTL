import unittest
import torch
from main import HungarianMatcher

class TestHungarianMatcher(unittest.TestCase):
    #matchers forward function takes outputs, tgt_sizes,tgt_embs,tgt_bbox):         

    def test_matcher_output_shape(self):
        matcher = HungarianMatcher(1, 1, 1)
        num_queries = 7
        num_assignments = 3
        num_targets = 7
        outputs={'pred_logits': torch.randn( 4,num_queries, 256),
                 'pred_boxes': torch.randn( 4,num_queries, 4),}

        # MAKE SURE WH ARE POSITIVE AND >XYXY
        outputs['pred_boxes'][:] = torch.abs(outputs['pred_boxes'][:])
        outputs['pred_boxes'][:,:,2:] += outputs['pred_boxes'][:,:,:2]
        tgt_sizes=torch.tensor( [1,2,3,1])
        tgt_embed=torch.randn(num_targets, 256)
        #out prob will be the batch*queries * f
        tgt_bbox=torch.randn(num_targets, 4)
        tgt_bbox= torch.abs(tgt_bbox)
        tgt_bbox[:,2:] += tgt_bbox[:,:2]
        output = matcher(outputs, tgt_sizes,tgt_embed,tgt_bbox)
        self.assertEqual([*output.shape], [3,num_queries])
        
    def test_matcher_output_values(self):
        matcher = HungarianMatcher(1, 1, 1)
        num_queries = 7
        num_assignments = 3
        num_targets = 7
        outputs={'pred_logits': torch.randn( 4,num_queries, 256),
                 'pred_boxes': torch.randn( 4,num_queries, 4),}

        # MAKE SURE WH ARE POSITIVE AND >XYXY
        outputs['pred_boxes'][:] = torch.abs(outputs['pred_boxes'][:])
        outputs['pred_boxes'][:,:,2:] += outputs['pred_boxes'][:,:,:2]
        tgt_sizes=torch.tensor( [1,2,3,1])
        tgt_embed=torch.randn(num_targets, 256)
        #out prob will be the batch*queries * f
        tgt_bbox=torch.randn(num_targets, 4)
        tgt_bbox= torch.abs(tgt_bbox)
        tgt_bbox[:,2:] += tgt_bbox[:,:2]
        output = matcher(outputs, tgt_sizes,tgt_embed,tgt_bbox)
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output < num_targets))
        
    def test_matcher_output_unique(self):
        matcher = HungarianMatcher(1, 1, 1)
        num_queries = 7
        num_assignments = 3
        num_targets = 7
        outputs={'pred_logits': torch.randn( 4,num_queries, 256),
                 'pred_boxes': torch.randn( 4,num_queries, 4),}

        # MAKE SURE WH ARE POSITIVE AND >XYXY
        outputs['pred_boxes'][:] = torch.abs(outputs['pred_boxes'][:])
        outputs['pred_boxes'][:,:,2:] += outputs['pred_boxes'][:,:,:2]
        tgt_sizes=torch.tensor( [1,2,3,1])
        tgt_embed=torch.randn(num_targets, 256)
        #out prob will be the batch*queries * f
        tgt_bbox=torch.randn(num_targets, 4)
        tgt_bbox= torch.abs(tgt_bbox)
        tgt_bbox[:,2:] += tgt_bbox[:,:2]
        output = matcher(outputs, tgt_sizes,tgt_embed,tgt_bbox)
        print(output)
        unique_output = torch.unique(output,dim=1)
        print(unique_output)
        self.assertEqual(output.shape, unique_output.shape)
import torch
import pytest
from main import FastCriterion
class TestFastCriterion(unittest.TestCase):
    def test_fast_criterion():
        # Create a dummy model output
        num_queries = 3
        batch_size = 2
        num_classes = 10
        num_boxes = 5
        model_output = {
            'pred_logits': torch.randn(batch_size, num_queries, num_classes),
            'pred_boxes': torch.randn(batch_size, num_queries, 4),
            'pred_masks': torch.randn(batch_size, num_queries, 1, 28, 28),
            'aux_outputs': [
                {'pred_logits': torch.randn(batch_size, num_queries, num_classes),
                'pred_boxes': torch.randn(batch_size, num_queries, 4)}
            ]
        }

        # Create a dummy target
        target = {
            'labels': torch.randint(num_classes, size=(batch_size, num_boxes)),
            'boxes': torch.randn(batch_size, num_boxes, 4),
            'masks': torch.randint(2, size=(batch_size, num_queries, 1, 28, 28))
        }

        # Create the criterion and compute the loss
        criterion = FastCriterion(1, 1, 1)
        loss_dict = criterion(model_output, target)

        # Check that the loss is a scalar tensor
        assert isinstance(loss_dict['loss'], torch.Tensor)
        assert loss_dict['loss'].shape == torch.Size([])

        # Check that the model learns from the loss
        optimizer = torch.optim.SGD(criterion.parameters(), lr=0.1)
        initial_loss = loss_dict['loss'].item()
        optimizer.zero_grad()
        loss_dict['loss'].backward()
        optimizer.step()
        new_loss_dict = criterion(model_output, target)
        new_loss = new_loss_dict['loss'].item()
        assert new_loss < initial_loss
    def test_set_criterion():
        # Create a dummy model
        model = torch.nn.Linear(10, 2)

        # Create some dummy inputs
        num_queries = 5
        num_classes = 3
        num_boxes = 10
        logits = torch.randn(num_queries, num_boxes, num_classes)
        targets = {
            'labels': torch.randint(0, num_classes, (num_queries, num_boxes)),
            'boxes': torch.randn(num_queries, num_boxes, 4),
        }

        # Create the criterion
        weight_dict = {'loss_out_iou': 1, 'class_loss': 1}
        criterion = SetCriterion(weight_dict=weight_dict, num_classes=num_classes, matcher=None)

        # Compute the loss and gradients
        loss = criterion(model(logits), targets)
        loss.backward()

        # Check that the loss is a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

        # Check that the gradients are not None
        for param in model.parameters():
            assert param.grad is not None

import torch
from main import SetCriterion


if __name__ == '__main__':
    unittest.main()