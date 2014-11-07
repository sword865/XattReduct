package macLn.src;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class k_modesExp2 {
	double[][] ac=new double[7][100];
	double[][] pe=new double[7][100];
	double[][] re=new double[7][100];
	double[][] nmi=new double[7][100];
	double[][] ari=new double[7][100];
	double[][] ms=new double[7][100];
	double[][] iters=new double[7][100];
	double[][] ofv=new double[7][100];
	double[][] times=new double[7][100];
	double[] rpara=new double[3];
	int k_clusters;
	int U;
	public void main(String path,String file,String es)throws Exception
	{
	   Instances data=loadFile(path+file+".arff");
	   
	   boolean dele=true;
	   String deleStr=null;
	   if(dele){
		   deleMiss(data);
		   deleStr="dele_miss";
	   }else{
		   replaceMiss(data);
		   deleStr="replace_miss";
	   }
	   
	   Scanner scanner=new Scanner(System.in);
	  // k_clusters=scanner.nextInt();
	   k_clusters=data.numClasses();
	   //System.out.println(k_clusters);
	   Random rnd=new Random();
	   
	   U=data.numInstances();
	   int[] klable=new int[U];
	   
	   long time1,time2;
	   for(int i=0;i<100;++i){
		   data.randomize(rnd);
		   time1=System.currentTimeMillis();
		   k_modes.kModes(data,k_clusters,klable,rpara,es);
		   time2=System.currentTimeMillis();
		   times[0][i]=time2-time1;
		   evaRes(data,klable,i,0);
		   //utility.setVec(klable,0);
		   time1=System.currentTimeMillis();
		   k_modesModify.kModes(data,k_clusters,klable,rpara,es);
		   evaRes(data,klable,i,1);
		   time2=System.currentTimeMillis();
		   times[1][i]=time2-time1;
		   
		   time1=System.currentTimeMillis();
		   k_modesWeight.kModes(data, k_clusters, klable,rpara,es);
		   evaRes(data,klable,i,2);
		   time2=System.currentTimeMillis();
		   times[2][i]=time2-time1;
		   
		   time1=System.currentTimeMillis();
		   k_modesFre2.kModes(data, k_clusters, klable,rpara,es);
		   evaRes(data,klable,i,3);
		   time2=System.currentTimeMillis();
		   times[3][i]=time2-time1;
		   
		   time1=System.currentTimeMillis();
		   k_modesModifyRe.kModes(data, k_clusters, klable,rpara,es);
		   evaRes(data,klable,i,4);
		   time2=System.currentTimeMillis();
		   times[4][i]=time2-time1;
		   
		   time1=System.currentTimeMillis();
		   k_modesImprove.kModes(data, k_clusters, klable,rpara,es);
		   evaRes(data,klable,i,5);
		   time2=System.currentTimeMillis();
		   times[5][i]=time2-time1;
		   
		   time1=System.currentTimeMillis();
		   k_modesFreRe.kModes(data, k_clusters, klable,rpara,es);
		   evaRes(data,klable,i,6);
		   time2=System.currentTimeMillis();
		   times[5][i]=time2-time1;
	   }
	   outputRes(data,file,deleStr,es);
	}
	
	public void outputRes(Instances data,String file,String deleStr,String es)throws Exception
	{
		String[] criteria={"ac","pe","re","nmi","ari","ms","iters","ofv","times"};
		
		outputCriteriaRes(data,file,deleStr,criteria[0],ac,es);	
		outputCriteriaRes(data,file,deleStr,criteria[1],pe,es);
		outputCriteriaRes(data,file,deleStr,criteria[2],re,es);
		outputCriteriaRes(data,file,deleStr,criteria[3],nmi,es);
		outputCriteriaRes(data,file,deleStr,criteria[4],ari,es);
		outputCriteriaRes(data,file,deleStr,criteria[5],ms,es);
		outputCriteriaRes(data,file,deleStr,criteria[6],iters,es);
		modifyBase(ofv);
		outputCriteriaRes(data,file,deleStr,criteria[7],ofv,es);
		outputCriteriaRes(data,file,deleStr,criteria[8],times,es);
	}
	public void modifyBase(double[][] v)
	{
		for(int i=1;i<v.length;++i){
			for(int j=0;j<100;++j){
				v[i][j]-=v[0][j];
			}
		}
		for(int j=0;j<100;++j){
			v[0][j]=0;
		}
	}
	public void outputCriteriaRes(Instances data,String file,String deleStr,String cri,double[][] rs,String es)throws Exception
	{
		DecimalFormat decimal = new DecimalFormat("#.###");
		FileWriter fileWriter=new FileWriter("E:\\result2\\"+es+"\\100\\"+file+"With"+cri+".txt");
		fileWriter.append(""+file+"\r\n");
		fileWriter.append(cri+"\r\n");
		fileWriter.append(deleStr+"\r\n");
		//fileWriter.append("numClusters£º "+String.valueOf(k_clusters)+"\r\n");
		int []key=new int[100];
		double[] v=rs[1].clone();
		utility.sortValues(v,key);
		for(int i=0;i<100;++i){
				fileWriter.append(""+i+" "+decimal.format(rs[0][key[i]])+" "+decimal.format(rs[1][key[i]])+" "+
						decimal.format(rs[2][key[i]])+" "+decimal.format(rs[3][key[i]])+" "+decimal.format(rs[4][key[i]])
						+" "+decimal.format(rs[5][key[i]])+" "+decimal.format(rs[6][key[i]])+"\r\n");
		}
		fileWriter.flush();
		fileWriter.close();
		
		FileWriter fileWriter2=new FileWriter("E:\\result2\\"+es+"\\sum\\"+cri+".txt",true);
		fileWriter2.append(""+decimal.format(eva.Mean(rs[0]))+" "+decimal.format(eva.Mean(rs[1]))+" "+decimal.format(eva.Mean(rs[2]))+
				  " "+decimal.format(eva.Mean(rs[3]))+" "+decimal.format(eva.Mean(rs[4]))+" "+
				  decimal.format(eva.Mean(rs[5]))+" "+decimal.format(eva.Mean(rs[6])));
		
		fileWriter2.append("\r\n");
		fileWriter2.flush();
		fileWriter2.close();
//		fileWriter2.append(""+eva.standardDev(rs[0])+" "+eva.standardDev(rs[1])+" "+eva.standardDev(rs[2])+" "
//				+eva.standardDev(rs[3])+" "+eva.standardDev(rs[4]));
//		fileWriter2.append("\r\n");
		//fileWriter.append("descriptor"+" x "+"km"+cri+" km2"+cri+" kmw"+cri+" kmf"+cri+"\r\n");
		//fileWriter.append("%k_modes==km  "+"k_modes2==km2  "+"k_modesWeight==kmw  "+"kmodesFrequent==kmf  "+"\r\n");
	}
	public  void evaRes(Instances data,int[] klable,int iter,int method)throws Exception
	{		
		  double rs[]=new double[6];
		  eva.evalution(data, klable, k_clusters, rs);
		  assignRs(rs,iter,method);
	}
	
	public  void assignRs(double[]rs,int iter,int method)
	{
		ac[method][iter]=rs[0];
		pe[method][iter]=rs[1];
		re[method][iter]=rs[2];
		nmi[method][iter]=rs[3];
		ari[method][iter]=rs[4];
		ms[method][iter]=rs[5];
		iters[method][iter]=rpara[0];
		ofv[method][iter]=rpara[1];
	}
	public static Instances loadFile(String filename)throws Exception
	{
		FileReader fr=new FileReader(filename);
		Instances data=new Instances(fr);
		data.setClassIndex(data.numAttributes()-1);
		System.out.println("lablenum:"+data.numClasses()+" "+data.numInstances());
		return data;
	}
	
	public static void replaceMiss(Instances data)throws Exception
	{
		ReplaceMissingValues replaceMissingValues_Filter = new ReplaceMissingValues();    //¸²¸Ç¿ÕÖµ
		replaceMissingValues_Filter.setInputFormat(data);                                //¸²¸Ç¿ÕÖµ
		Instances filteredInstances = Filter.useFilter(data, replaceMissingValues_Filter);//¸²¸Ç¿ÕÖµ
		data=filteredInstances;
	}
	
	public static void deleDulp(Instances data)
	{
		//System.out.print(data.instance(0).);
	}
	public static void deleMiss(Instances data)
	{
		for(int i=0;i<data.numAttributes();++i){
			data.deleteWithMissing(i);
		}
		//data.deleteAttributeAt(0);
	}
}
