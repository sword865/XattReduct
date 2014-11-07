package macLn.src;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import weka.core.Instance;
import weka.core.Instances;

public class k_modesWeight {
	public static void main(String[] args)throws Exception
	{
	 
	}
	public static void kModes(Instances data,int k,int[] klable,double[] runPara,String es) throws IOException
	{
		Instance[] centers=new Instance[k];
		Instance[] centers2=new Instance[k];
		for(int i=0;i<k;++i){
			centers[i]=(Instance) data.instance(i).copy();
			centers2[i]=(Instance)data.instance(i).copy();
		}
		
		FileWriter fileWriter=new FileWriter("E:\\result2\\"+es+"\\iters\\RunTimeParakw"+".txt",true);
		double minObjectValue=1000000000;
		double F;
		int count=0;
		k_modes.assign(data,centers,klable);             //把样本分配到不同中心
		while(true){
			count++;
			//调整中心
			adjustCentersNew(data,centers,centers2,klable);
		//判断是否停止
			F=objectFunction(data,centers,centers2,klable);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong1:at kmodesWeight F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
			assignNew(data,centers,centers2,klable);             //把样本分配到不同中心
				
			//判断是否停止
			F=objectFunction(data,centers,centers2,klable);
			fileWriter.append(""+F+" ");
			if(F == minObjectValue){
				break;
			}else if(F > minObjectValue){
				//System.out.println("wrong2:at kmodesWeight F:"+F+"  "+minObjectValue );
				break;
			}else{
				minObjectValue=F;
			}
		
		}
		fileWriter.append("\r\n");
		fileWriter.flush();
		fileWriter.close();
		runPara[0]=count;
		runPara[1]=k_modes.objectFunction(data,centers,klable);
		//System.out.println("kmodesWeight done");
	}
	
	
	public static double objectFunction(Instances data,Instance[] centers,Instance[] centers2,int[] klable)
	{
		double gobalDis=0;
		int U=data.numInstances();
		for(int i=0;i<U;++i){
			gobalDis+= disNew(i,klable[i],data,centers,centers2);
		}
		return gobalDis;
	}
	
	//样本到中心的距离   原始kmode的公式  attr全是categories
	public static double disNew(int p,int z,Instances data,Instance[] centers,Instance[] centers2)
	{
		double distance=0;
	      //如果有类标则不算类标，所以减一 
		double weight=0.5;
		for(int i=0;i<data.numAttributes()-1;++i){
			if(data.instance(p).value(i)!=centers[z].value(i)){
				distance+=weight;
			}
			if(data.instance(p).value(i)!=centers2[z].value(i)){
				distance+= 1-weight;
			}
		}
		return distance;
	}
	public static void assignNew(Instances data,Instance[] centers,Instance[] centers2,int[] klable)
	{
		int U=data.numInstances();
		int k=centers.length;
		for(int i=0;i<U;++i){
			double mindis=5000000;
			for(int z=0;z<k;++z){
				double d=disNew(i,z,data,centers,centers2);
				if(d<mindis){
					mindis=d;
					klable[i]=z;
				}
			}
		}
	}
	public static void adjustCentersNew(Instances data,Instance[] centers,Instance[] centers2,int[] klable)
	{
		int k=centers.length;
		for(int z=0;z<k;++z){
		    for(int j=0;j<data.numAttributes()-1;++j){
		    	double[] d=utility.maxCate(data,klable,z,j);
		    	centers[z].setValue(j, d[0]);
		    	centers2[z].setValue(j, d[1]);	
			}
		}
	}
	
//	public static double[] maxCateNew(Instances data,int[] klable,int z,int j)
//	{
//		int U=data.numInstances();
//    	int[] cate=new int[50];
//    	for(int i=0;i<U;++i){
//    		if(klable[i]==z){
//    			double s=data.instance(i).value(j);
//    			if(s>50){
//    				System.out.print(data.attribute(j));
//    			}
//    			cate[(int)(s)]++;
//			}	
//		}
//    	double maxnum=0,num=0;
//    	double[] maxcate=new double[2];
//    	maxcate[0]=maxcate[1]=-1;
//    	for(int i=0;i<50;++i){
//    		if(cate[i]>0){
//    			num++;
//    		}
//    		if(cate[i]>maxnum){
//    			maxcate[0]=i;
//    			maxnum=cate[i];
//    		}
//    	}
//    	if(num<4){//data.attribute(j).numValues()<4){
//    		maxcate[1]=maxcate[0];
//    	}else{
//    		maxnum=-1;
//	    	for(int i=0;i<50;++i){
//	    		if(cate[i]>maxnum && i!=maxcate[0]){
//	    			maxcate[1]=i;
//	    			maxnum=cate[i];
//	    		}
//	    	}
//    	}
//    	return maxcate;
//	}
//	
//	
}

