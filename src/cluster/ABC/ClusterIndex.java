package cluster.ABC;

import weka.core.Instances;

public class ClusterIndex implements TargetFun{

	Instances dataset;
	
	public ClusterIndex(Instances inputdata){
		dataset=new Instances(inputdata);		
	}
	
	public void setDateset(Instances inputdata){
		dataset=new Instances(inputdata);
	}
	
	@Override
	public double getfitnessvalue(double[] input) {
		// TODO Auto-generated method stub
		return 0;
	}
	
	
	
}
