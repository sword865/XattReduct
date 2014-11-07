package cluster.ABC;

import weka.core.Instances;

public class Jm extends ClusterIndex{
	
	
	public Jm(Instances inputdata) {
		super(inputdata);
		// TODO Auto-generated constructor stub
	}

	@Override
	public double getfitnessvalue(double[] input) {
		int cluster_k=input.length/dataset.numAttributes();
		int[] idx=new int[dataset.numInstances()];
		double[] neardistances=new double[dataset.numInstances()];
		for(int i=0;i<idx.length;i++){
			double mindisvalue=Double.MAX_VALUE;int sel=-1;
			for(int j=0;j<cluster_k;j++){
				double tmpdis=0;
				for(int k=0;k<dataset.numAttributes();k++){
					tmpdis+=Math.pow(dataset.instance(i).value(k)-input[j*dataset.numAttributes()+k] , 2);
				}
				if(tmpdis<mindisvalue){
					mindisvalue=tmpdis;
					sel=j;
				}
			}
			idx[i]=sel;
			neardistances[i]=mindisvalue;
		}				
		double Jm=0;
		for(int i=0;i<dataset.numInstances();i++){
			Jm+=neardistances[i];
		}
		return Jm;
	}

}
