import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime
import pandas as pd
import argparse

class FakeDetectionEvaluator:
    def __init__(self, methods, base_pred_path, gt_path, context_data_path, inpainting_info_path, all_masks_path, enhancement_factor=1.3):
        """
        Initialize the evaluator
        Args:
            methods: List of methods to evaluate
            base_pred_path: Base path for predictions
            gt_path: Path to ground truth masks
            context_data_path: Path to context_data.csv
            inpainting_info_path: Path to inpainting_info.csv
            all_masks_path: Path to all_masks directory
            enhancement_factor: Factor to enhance prediction values (default: 1.3)
        """
        self.methods = methods
        self.base_pred_path = base_pred_path
        self.gt_path = gt_path
        self.enhancement_factor = enhancement_factor
        self.pred_paths = {
            method: os.path.join(base_pred_path, f"{method}_predictions")
            for method in methods
        }
        
        # Load context and inpainting data
        self.context_data = pd.read_csv(context_data_path)
        self.inpainting_info = pd.read_csv(inpainting_info_path)
        self.all_masks_path = all_masks_path
        
    def load_prediction(self, img_path):
        """
        Load prediction image saved by plt.imsave
        """
        return np.load(img_path)
    
    def load_ground_truth(self, gt_path):
        """
        Load ground truth mask using PIL
        """
        mask = np.array(Image.open(gt_path).convert("L"))
        mask = mask > 0
        return mask.astype(np.float32)

    def enhance_prediction(self, pred, img_name):
        """
        Enhance prediction based on context and inpainting information
        """
        try:
            # Get coco_index from image name
            coco_index = int(os.path.splitext(img_name)[0])
            
            # Check if this index exists in context data and has label 1
            context_row = self.context_data[self.context_data['coco_index'] == coco_index]
            if not context_row.empty and context_row['label'].iloc[0] == 1:
                # Get replacement object
                inpainting_row = self.inpainting_info[self.inpainting_info['coco_index'] == coco_index]
                if not inpainting_row.empty:
                    replacement_object = inpainting_row['replacement_object'].iloc[0]
                    
                    # Load object mask
                    mask_path = os.path.join(self.all_masks_path, str(coco_index), f"{replacement_object}.png")
                    if os.path.exists(mask_path):
                        object_mask = np.array(Image.open(mask_path).convert("L")) > 0
                        # Apply enhancement
                        enhanced_pred = pred.copy()
                        enhanced_pred[object_mask] = np.minimum(
                            enhanced_pred[object_mask] * self.enhancement_factor,
                            1.0
                        )
                        # print(f"coco index is {coco_index}")
                        return enhanced_pred
            
            return pred
            
        except Exception as e:
            print(f"Error enhancing prediction for {img_name}: {str(e)}")
            return pred
    
    def evaluate_single_image(self, pred_path, gt_path, img_name, threshold=0.5):
        """
        Evaluate metrics for a single image
        """
        try:
            pred = self.load_prediction(pred_path)
            # print(pred.shape)
            gt = self.load_ground_truth(gt_path)
            # print(gt.shape)
            
            # Skip if ground truth is all ones or zeros
            if np.all(gt == 1) or np.all(gt == 0):
                return None
            
            # Enhance prediction based on context
            pred = self.enhance_prediction(pred, img_name)
            
            # Calculate binary metrics
            pred_binary = pred > threshold
            accuracy = accuracy_score(gt.ravel(), pred_binary.ravel())
            # print(f"image name is {img_name}, accuracy is {accuracy}")
            f1 = f1_score(gt.ravel(), pred_binary.ravel())
            
            # Calculate threshold-free metrics
            try:
                auc = roc_auc_score(gt.ravel(), pred.ravel())
                ap = average_precision_score(gt.ravel(), pred.ravel())
            except ValueError:
                auc = np.nan
                ap = np.nan
                
            return {
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'ap': ap
            }
        except Exception as e:
            print(f"Error processing {pred_path}: {str(e)}")
            return None
    
    def visualize_examples(self, num_examples=5, save_dir=None):
        """
        Visualize predictions and ground truth for multiple images
        """
        # Get available predictions for the first method
        pred_files = [f for f in os.listdir(self.pred_paths[self.methods[0]]) 
                     if f.endswith('.png')][:num_examples]
        
        for img_name in pred_files:
            gt_path = os.path.join(self.gt_path, img_name)
            if not os.path.exists(gt_path):
                continue
                
            fig, axes = plt.subplots(1, len(self.methods) + 1, figsize=(5*(len(self.methods) + 1), 4))
            
            # Show ground truth
            gt = self.load_ground_truth(gt_path)
            axes[0].imshow(gt, cmap='gray')
            axes[0].set_title('Ground Truth')
            axes[0].axis('off')
            
            # Show predictions
            for i, method in enumerate(self.methods):
                pred_path = os.path.join(self.pred_paths[method], img_name)
                pred = self.load_prediction(pred_path)
                # Enhance prediction
                pred = self.enhance_prediction(pred, img_name)
                
                axes[i+1].imshow(pred, cmap='RdBu_r', vmin=0, vmax=1)
                axes[i+1].set_title(f'{method}')
                axes[i+1].axis('off')
            
            plt.suptitle(f'Image: {img_name}')
            plt.tight_layout()
            
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'visualization_{img_name}'))
                plt.close()
            else:
                plt.show()
    
    def evaluate_all(self, max_samples=None, save_results=True):
        """
        Evaluate all methods on all available predictions
        """
        results = {method: {
            'accuracy': [], 'f1': [], 'auc': [], 'ap': [],
            'skipped_images': [],
            'missing_gt_images': [],
            'missing_pred_images': [],
            'error_images': []
        } for method in self.methods}
        
        # Get available predictions for the first method
        # Prediction results are npy file
        pred_files = [f for f in os.listdir(self.pred_paths[self.methods[0]]) 
                     if f.endswith('.npy')]
        
        if max_samples:
            pred_files = pred_files[:max_samples]
        ## masks are png files
        for img_name in tqdm(pred_files, desc="Evaluating images"):
            mask_img_name = img_name.replace('.npy', '.png')
            gt_path = os.path.join(self.gt_path, mask_img_name)
            
            if not os.path.exists(gt_path):
                print(f"Ground truth not found: {gt_path}")
                for method in self.methods:
                    results[method]['missing_gt_images'].append(img_name)
                continue
                
            for method in self.methods:
                pred_path = os.path.join(self.pred_paths[method], img_name)
                if not os.path.exists(pred_path):
                    print(f"Prediction not found for {method}: {pred_path}")
                    results[method]['missing_pred_images'].append(img_name)
                    continue
                    
                metrics = self.evaluate_single_image(pred_path, gt_path, img_name)
                
                if metrics is None:
                    results[method]['skipped_images'].append(img_name)
                    continue
                    
                for metric_name, value in metrics.items():
                    results[method][metric_name].append(value)
        
        # Calculate average metrics and organize results
        final_results = {}
        for method in self.methods:
            # Calculate average metrics
            final_results[method] = {
                metric: float(np.mean(values)) for metric, values in results[method].items()
                if isinstance(values, list) and values and metric not in ['skipped_images', 'missing_gt_images', 'missing_pred_images', 'error_images']
            }
            
            # Add skipped images information
            final_results[method]['skipped_images_count'] = len(results[method]['skipped_images'])
            final_results[method]['skipped_images_list'] = results[method]['skipped_images']
            
            # Add missing images information
            final_results[method]['missing_gt_count'] = len(results[method]['missing_gt_images'])
            final_results[method]['missing_gt_list'] = results[method]['missing_gt_images']
            final_results[method]['missing_pred_count'] = len(results[method]['missing_pred_images'])
            final_results[method]['missing_pred_list'] = results[method]['missing_pred_images']
            
            # Add total processed images
            final_results[method]['total_processed'] = len(results[method]['accuracy'])
            final_results[method]['total_attempted'] = (len(results[method]['accuracy']) + 
                                                      len(results[method]['skipped_images']) +
                                                      len(results[method]['missing_gt_images']) +
                                                      len(results[method]['missing_pred_images']))
        
        if save_results:
            self.save_results(final_results)
            
        return final_results
    
    def save_results(self, results):
        """
        Save evaluation results to a JSON file
        """
        # Create evaluation_results directory if it doesn't exist
        save_dir = f"evaluation_results/oracle/{self.enhancement_factor}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"results.json")
        
        # Add metadata to results
        results_with_metadata = {
            "metadata": {
                "timestamp": timestamp,
                "base_pred_path": self.base_pred_path,
                "gt_path": self.gt_path,
                "methods": self.methods,
                "enhancement_factor": self.enhancement_factor
            },
            "results": results
        }
        
        # Save to JSON file
        with open(filename, 'w') as f:
            json.dump(results_with_metadata, f, indent=4)
        
        print(f"Results saved to: {filename}")

# Usage example:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a binary classification model.")
    parser.add_argument('--gamma', type=int, default=5,
                       help="enhancement factor to use (e.g., 1, 2, 5, 10 etc.)")
    
    args = parser.parse_args()

    base_pred_path = "../task_data/fake_localization/baselines"
    gt_path = "../task_data/masks/bbox_masks_testing"
    context_data_path = "../task_data/fake_localization/context_data.csv"
    inpainting_info_path = "../task_data/fake_localization/inpainting_info.csv"
    all_masks_path = "../task_data/fake_localization/all_masks"
    methods = ['CAT-Net', 'ManTraNet', 'Trufor', "PSCC-Net"]
    evaluator = FakeDetectionEvaluator(
        methods, 
        base_pred_path, 
        gt_path,
        context_data_path,
        inpainting_info_path,
        all_masks_path,
        enhancement_factor=args.gamma
    )

    # Evaluate images and save results
    results = evaluator.evaluate_all()

    # Print results
    for method, metrics in results.items():
        print(f"\n{method} Results:")
        print(f"Performance Metrics:")
        for metric_name in ['accuracy', 'f1', 'auc', 'ap']:
            if metric_name in metrics:
                print(f"{metric_name}: {metrics[metric_name]:.4f}")

