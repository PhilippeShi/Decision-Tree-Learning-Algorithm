import java.text.*;
import java.lang.Math;
 
public class DecisionTree implements Serializable {
 
	DTNode rootDTNode;
	int minSizeDatalist; //minimum number of datapoints that should be present in the dataset so as to initiate a split

	public static final long serialVersionUID = 343L;
	
	public DecisionTree(ArrayList<Datum> datalist , int min) {
		minSizeDatalist = min;
		rootDTNode = (new DTNode()).fillDTNode(datalist);
	}
 
	class DTNode implements Serializable{
		//Mention the serialVersionUID explicitly in order to avoid getting errors while deserializing.
		public static final long serialVersionUID = 438L;
		boolean leaf;
		int label = -1;      // only defined if node is a leaf
		int attribute; // only defined if node is not a leaf
		double threshold;  // only defined if node is not a leaf
 
		DTNode left, right; //the left and right child of a particular node. (null if leaf)
 
		DTNode() {
			leaf = true;
			threshold = Double.MAX_VALUE;
		}
 
		
		// this method takes in a datalist (ArrayList of type datum). It returns the calling DTNode object 
		// as the root of a decision tree trained using the datapoints present in the datalist variable and minSizeDatalist.
		DTNode fillDTNode(ArrayList<Datum> datalist) {
			
			if (datalist.size() >= minSizeDatalist) {
				//check if all labels are the same
				boolean checkAllSame = true;
				for (int i = 0; i<datalist.size(); i++) {
					if(datalist.get(i).y != datalist.get(0).y) { //if all labels are same
						checkAllSame = false;
						break;
					}
				}
				//if all labels are the same, create leaf node
				if (checkAllSame) {
					//Don't need to set boolean leaf = true since it's the default value
					DTNode leaf = new DTNode();
					leaf.label = datalist.get(0).y;;
					return leaf;
				}
				
				//if labels are not all the same, have to split data
				else {
					//Calling helper method to get best attribute and threshold
					int bestAttr = (int) getBestAttrAndThr(datalist)[0];
					double bestThr =  getBestAttrAndThr(datalist)[1];
					
					//Creating new node
					DTNode node = new DTNode();
					node.attribute = bestAttr; node.threshold = bestThr; node.leaf = false;
					
					//Initializing splits
					ArrayList<Datum> leftS = new ArrayList<Datum>();
					ArrayList<Datum> rightS = new ArrayList<Datum>();
					
					//Separating data into the two splits
					for(Datum data : datalist) {
						if(data.x[bestAttr] < bestThr) leftS.add(data);
						else rightS.add(data);
					}
					//Setting left and right children nodes
					node.left = fillDTNode(leftS);
					node.right = fillDTNode(rightS);
					return node;
				}
			}
			
			else {
				DTNode leaf = new DTNode();
				leaf.label = findMajority(datalist);
				return leaf;
			}
		}
		/**
		 * Helper method that finds the best attribute and the best threshold to split a given datalist
		 * @param datalist (arrayList of type Datum)
		 * @return array (Attribute @index 0, Threshold @index 1)
		 */
		private double[] getBestAttrAndThr(ArrayList<Datum> datalist) {
			double bestAvgEntropy = Double.MAX_VALUE;
			int bestAttr = -1;
			double bestThr = -1;
			double currAvgEntropy = -1;
			
			//for all attributes (all datum have same amount of attributes) and for all datum,
			//find the entropy when splitting at all values of attributes
			for(int i = 0; i<datalist.get(0).x.length; i++) {
				for(int j = 0; j<datalist.size(); j++) {
					double threshold = datalist.get(j).x[i];
					ArrayList<Datum> leftSplit = new ArrayList<Datum>();
					ArrayList<Datum> rightSplit = new ArrayList<Datum>();
					for(Datum data : datalist) {
						if(data.x[i] < threshold) leftSplit.add(data);
						else rightSplit.add(data);
						
					}
					currAvgEntropy = calcEntropy(rightSplit) * rightSplit.size()/(rightSplit.size() + leftSplit.size()) + 
							calcEntropy(leftSplit) * leftSplit.size()/(rightSplit.size() + leftSplit.size());
					if(currAvgEntropy < bestAvgEntropy) {
						bestAvgEntropy = currAvgEntropy;
						bestThr = threshold;
						bestAttr = i;
					}
				}
			}
			return new double[] {bestAttr, bestThr};
		}
		
		// This is a helper method. Given a datalist, this method returns the label that has the most
		// occurrences. In case of a tie it returns the label with the smallest value (numerically) involved in the tie.
		int findMajority(ArrayList<Datum> datalist) {
			
			int [] votes = new int[2];
 
			//loop through the data and count the occurrences of datapoints of each label
			for (Datum data : datalist)
			{
				votes[data.y]+=1;
			}
			
			if (votes[0] >= votes[1])
				return 0;
			else
				return 1;
		}
		
		// This method takes in a datapoint (excluding the label) in the form of an array of type double (Datum.x) and
		// returns its corresponding label, as determined by the decision tree
		int classifyAtNode(double[] xQuery) {
			//Reached a leaf node, return the label
			if (this.leaf) return this.label;
			
			//Reached an internal node
			else {
				//Go to left child node when attribute < threshold
				if (xQuery[this.attribute] < this.threshold) {
					return this.left.classifyAtNode(xQuery);
				}
				//Go to right child node when attribute >= threshold
				else return this.right.classifyAtNode(xQuery);	
			}
		}
 
 
		//given another DTNode object, this method checks if the tree rooted at the calling DTNode is equal to the tree rooted
		//at DTNode object passed as the parameter
		public boolean equals(Object dt2)
		{ 
			if (!(dt2 instanceof DTNode)) return false;
			//Both null
			if (this == null && dt2 == null) return true;
			//Both non-null
			if (this != null && dt2 !=null ) {
				//Both leaves
				if (this.leaf && ((DTNode)dt2).leaf) return this.label == ((DTNode)dt2).label;
				//Both internal nodes
				if (!this.leaf && !((DTNode)dt2).leaf) 
					return (this.attribute == ((DTNode)dt2).attribute && this.threshold == ((DTNode)dt2).threshold && 
					this.left.equals(((DTNode)dt2).left) && this.right.equals(((DTNode)dt2).right));
				//Leaf & internal node
				else return false;	
			}
			//Null & non-null
			return false; 
		}
	}
 
 
 
	//Given a dataset, this returns the entropy of the dataset
	double calcEntropy(ArrayList<Datum> datalist) {
		double entropy = 0;
		double px = 0;
		float [] counter= new float[2];
		if (datalist.size()==0)
			return 0;
		double num0 = 0.00000001,num1 = 0.000000001;
 
		//calculates the number of points belonging to each of the labels
		for (Datum d : datalist)
		{
			counter[d.y]+=1;
		}
		//calculates the entropy
		for (int i = 0 ; i< counter.length ; i++)
		{
			if (counter[i]>0)
			{
				px = counter[i]/datalist.size();
				entropy -= (px*Math.log(px)/Math.log(2));
			}
		}
 
		return entropy;
	}
 
 
	// given a datapoint (without the label) calls the DTNode.classifyAtNode() on the rootnode of the calling DecisionTree object
	int classify(double[] xQuery ) {
		return this.rootDTNode.classifyAtNode( xQuery );
	}
 
	
	String checkPerformance( ArrayList<Datum> datalist) {
		DecimalFormat df = new DecimalFormat("0.000");
		float total = datalist.size();
		float count = 0;
 
		for (int s = 0 ; s < datalist.size() ; s++) {
			double[] x = datalist.get(s).x;
			int result = datalist.get(s).y;
			if (classify(x) != result) {
				count = count + 1;
			}
		}
 
		return df.format((count/total));
	}
 
 
	//Given two DecisionTree objects, this method checks if both the trees are equal by
	//calling onto the DTNode.equals() method
	public static boolean equals(DecisionTree dt1,  DecisionTree dt2)
	{
		boolean flag = true;
		flag = dt1.rootDTNode.equals(dt2.rootDTNode);
		return flag;
	}
 
}
