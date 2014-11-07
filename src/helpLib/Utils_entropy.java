package helpLib;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

//import Xreducer_gui.resultFrame;
//import Xreducer_struct.oneAlgorithm.xStyle;
import weka.core.Instances;
import weka.filters.Filter;

public class Utils_entropy {
	
	//求等价类Xi
	//boolean[]X 包括决策属性
	//暂不考虑missing值
	public static boolean[][] getEquivalenceClass(Instances dataset, boolean[] X){

		int U = dataset.numInstances();
		//求等价类,先求等价矩阵
		int[][] X_matrix = new int[U][U];
		int N = 0; //统计等价类的个数
		for(int i=0;i<U;++i){
			if(X_matrix[i][i]==0){
				N++;
				X_matrix[i][i] = 1;
				for(int j=i+1;j<U;++j){
					boolean flag = true;
					for(int k=0;k<X.length;++k){
						if(X[k]&&dataset.instance(i).value(k)!=dataset.instance(j).value(k)){
							flag = false;
							break;
						}
					}
					if (flag){
						X_matrix[i][j] = 1;
						X_matrix[j][j] = 2;
					}
				}
			}
		}
		//生成等价类Xi
		boolean[][] Xi = new boolean[U][N];
		int n=0;
		for (int i=0;i<U;++i){
			if(X_matrix[i][i]==1){
				for (int j=0;j<U;++j){
					if(X_matrix[i][j]==1)
					Xi[j][n] = true ;
				}
				n++;
			}
		}
		return Xi;
	}
	
	//得到给定属性子集B的粒度（granulation）(正域)POS_D(B)  POS_bD
	//其中B包括决策属性 pos=(U pos_By)/|U|

	public static double getGranulation(Instances dataset,boolean[] D, boolean[] B){
		boolean[][] Xi = getEquivalenceClass(dataset, B); //条件属性的等价类
		boolean[][] Yj = getEquivalenceClass(dataset, D); //决策属性的等价类
		
		int N = Xi[0].length;
		int U = Xi.length;
		int M = Yj[0].length;		
		//计算粒度
		//求POS(B) ,pos_sum 为|POS(B)|
		int pos_sum = 0;
		for(int i=0;i<N;++i){
			for(int j=0;j<M;++j){
				int cnt = 0;
				for(int k=0;k<U;++k){
					if(Xi[k][i] && Yj[k][j]){
						cnt++; //Xi \in Yj 的个数 [x]_B\in Yj
					}
					if(Xi[k][i] && !Yj[k][j]){//不属于
						cnt=0;
						break;
					}
				}
				pos_sum = pos_sum + cnt;
			}
		}
		return (double)pos_sum/U;	
//		int tmpsum = 0;
//		boolean[] tmp = new boolean[U];
//		for(int i=0;i<N;++i){
//			for(int j=0;j<M;++j){
//				int cnt = 0;
//				for(int k=0;k<U;++k){
//					if(Xi[k][i] && Yj[k][j]){
//						cnt++; //Xi \in Yj 的个数 [x]_B\in Yj
//					}
//					if(Xi[k][i] && !Yj[k][j]){//不属于
//						cnt=0;
//						break;
//					}
//				}
//				for(int k=0;k<U;++k){
//					if(Xi[k][i])
//						tmp[k]=true;
//				}
// 
//			}
//		}
//		for(int k=0;k<U;++k){
//			if(tmp[k])
//				tmpsum++;
//		}
//		return (double)tmpsum/U;		
	}
	//POS_pX
	public static boolean[] getPOS_pX(Instances dataset,boolean[] X, boolean[] P){
		boolean[][] Xi = getEquivalenceClass(dataset, P); 
		boolean[][] Yj = getEquivalenceClass(dataset, X); 
		int N = Xi[0].length;
		int U = Xi.length;
		int M = Yj[0].length;
		
		boolean[] pos_px =  new boolean[U];
		for(int i=0;i<N;++i){
			for(int j=0;j<M;++j){
				boolean[] px = new boolean[U];
				for(int k=0;k<U;++k){
					if(Xi[k][i] && Yj[k][j]){
						px[k]=true;
					}
					if(Xi[k][i] && !Yj[k][j]){ //不属于
						Arrays.fill(px, false);
						break;
					}					
				}
				pos_px = Utils.boolsAdd(pos_px, px);
			}
		}
		return pos_px;
	}
	//BND_pQ
	public static boolean[] getBND_pX(Instances dataset,boolean[] X, boolean[] P){
		boolean[][] Xi = getEquivalenceClass(dataset, P); 
		boolean[][] Yj = getEquivalenceClass(dataset, X); 
		int N = Xi[0].length;
		int U = Xi.length;
		int M = Yj[0].length;
		
		boolean[] L_px =  new boolean[U];
		for(int i=0;i<N;++i){
			for(int j=0;j<M;++j){
				boolean[] px = new boolean[U];
				for(int k=0;k<U;++k){
					if(Xi[k][i] && Yj[k][j]){
						px[k]=true;
					}
					if(Xi[k][i] && !Yj[k][j]){ //不属于
						Arrays.fill(px, false);
						break;
					}					
				}
				L_px = Utils.boolsAdd(L_px, px);
			}
		}
		boolean[] U_px =  new boolean[U];
		for(int i=0;i<N;++i){
			for(int j=0;j<M;++j){
				for(int k=0;k<U;++k){
					if(Xi[k][i] && Yj[k][j]){
						U_px = Utils.boolsAdd(U_px, Utils.boolean2oneEquivalence(Xi, i));
						break;
					}			
				}
			}
		}
		return Utils.boolsSubtract(U_px, L_px);
	}
	public static double[] getPOS_pmean(Instances dataset,boolean[] pos,boolean[] X, boolean[] P){
		int N = X.length;
		double[] Pmean = new double[N];
		int numpos = Utils.booleanSelectedNum(pos);
		if(numpos>=1){
			for(int i=0;i<N;++i){
				if(P[i]){//第i个属性
					double[] Temppx = new double[numpos];
					int p = 0;
					for(int k=0;k<pos.length;++k){
						if(pos[k]){ //k在正域内
							//sumpx = sumpx + dataset.instance(k).value(i);
							//System.out.println(k+","+i+":"+dataset.instance(k).value(i));
							Temppx[p++] = dataset.instance(k).value(i);
						}
					}
					Pmean[i] = Utils.PosmeanOperater(Temppx);
				}
			}
		}
		return Pmean;
	}
	public static double getDistanceMeasure_pQ(Instances dataset,boolean[] Q, boolean[] P){
		double WpQ = 0.0;
		boolean[] pos =  getPOS_pX(dataset,Q,P);
		//System.out.println(Arrays.toString(pos));
		if(Utils.isAllFalse(pos))
			return 0.0;
		if(Utils.isAllTrue(pos))
			return 1.0;
		double RpQ = getGranulation(dataset,Q,P);
		double[] Pmean = getPOS_pmean(dataset,pos,Q,P);
		//System.out.println(Arrays.toString(Pmean));
		boolean[] BND_pQ = getBND_pX(dataset,Q,P);
		double delta = 0.0;
		int U = dataset.numInstances();
		int N = dataset.numAttributes();
		for(int i=0;i<U;++i){
			if(BND_pQ[i]){  // i\in BND
				double delta_k = 0.0;
				for(int k=0;k<N;++k){
					if(P[k]){
						double fa = Utils.DistanceMetrics(Pmean[k], dataset.instance(i).value(k));
						delta_k = delta_k + Math.pow(fa,2);
					}
				}
				//System.out.println(i+":"+Math.sqrt(delta_k));
				delta = delta + Math.sqrt(delta_k);
			}
		}
		WpQ = delta!=0.0?1.0/delta:0.0;
		return (WpQ+RpQ)/(double)2;
		//return WpQ;
	}
	//求信息熵 information entropy
	//H(X) boolean[]X 包括决策属性
	public static double getInformationEntorpy(Instances dataset, boolean[] X){
		boolean[][] Xi = getEquivalenceClass(dataset, X);
		int U = Xi.length;
		int N = Xi[0].length;
		double res_XIH = 0.0;
		for(int i=0;i<N;++i){
			int sum=0;
			for(int k=0;k<U;++k){
				if(Xi[k][i])
				sum++;
			}
			if(sum!=(double)0){
				//System.out.println(sum+"/"+U);
				res_XIH=res_XIH-(sum/(double)U)* (Math.log((double)sum/U) / Math.log((double)2));
				//System.out.println((Math.log((double)sum/U) / Math.log((double)2)));
				//System.out.println(-(sum/(double)U)*(Math.log((double)sum/U) / Math.log((double)2)));
				//System.out.println(res_XIH);
			}
			
		}
		//res_XIH=(double)res_XIH/(double)U;
		return res_XIH;
	}
	

	
	public static boolean[][][] getEquivalenceClassforSemi(Instances dataset, boolean[] X){
		boolean[] labvec=new boolean[dataset.numInstances()];
		//Instances ldata=new Instances(dataset,0);
		for(int i=0;i<dataset.numInstances();i++){
			if(!dataset.instance(i).classIsMissing()){
				labvec[i]=true;
				//ldata.add(dataset.instance(i));
			}else{
				labvec[i]=false;
			}
		}
		int U = dataset.numInstances();
		//ldata.numInstances();
		//求等价类,先求等价矩阵
		int[][] X_matrix = new int[U][U];
		int N = 0; //统计等价类的个数
		for(int i=0;i<U;++i){
			if(X_matrix[i][i]==0){
				N++;
				X_matrix[i][i] = 1;
				for(int j=i+1;j<U;++j){
					boolean flag = true;
					for(int k=0;k<X.length;++k){
						if(X[k]&&dataset.instance(i).value(k)!=dataset.instance(j).value(k)){
							flag = false;
							break;
						}
					}
					if (flag){
						X_matrix[i][j] = 1;
						X_matrix[j][j] = 2;
					}
				}
			}
		}
		//生成等价类Xi
		boolean[][][] Xi = new boolean[2][U][N];
		int n=0;
		for (int i=0;i<U;++i){
			if(X_matrix[i][i]==1){
				for (int j=0;j<U;++j){
					if(X_matrix[i][j]==1){
						Xi[0][j][n] = true ;
						if(labvec[j]){
							Xi[1][j][n]=true;
						}
					}
				}
				n++;
			}
		}
		return Xi;
	}
	
	public static boolean[][] getEquivalenceClassforlabel(Instances dataset, boolean[] X){
		boolean[] labvec=new boolean[dataset.numInstances()];
		int U = dataset.numInstances();
		//求等价类,先求等价矩阵
		int[][] X_matrix = new int[U][U];
		int N = 0; //统计等价类的个数
		for(int i=0;i<U;++i){
			if(dataset.instance(i).classIsMissing())
				continue;
			if(X_matrix[i][i]==0){
				N++;
				X_matrix[i][i] = 1;
				for(int j=i+1;j<U;++j){
					if(dataset.instance(j).classIsMissing())
						continue;
					boolean flag = true;
					for(int k=0;k<X.length;++k){
						if(X[k]&&dataset.instance(i).value(k)!=dataset.instance(j).value(k)){
							flag = false;
							break;
						}
					}
					if (flag){
						X_matrix[i][j] = 1;
						X_matrix[j][j] = 2;
					}
				}
			}
		}
		//生成等价类Xi
		boolean[][] Xi = new boolean[U][N];
		int n=0;
		for (int i=0;i<U;++i){
			if(X_matrix[i][i]==1){
				for (int j=0;j<U;++j){
					if(X_matrix[i][j]==1)
					Xi[j][n] = true ;
				}
				n++;
			}
		}
		return Xi;
	}
	public static double getSemiJointEntropy(Instances dataset, boolean[] X, boolean[] Y){
		boolean[][][] Yj= getEquivalenceClassforSemi(dataset, Y);	
		boolean[][] Xi = getEquivalenceClassforlabel(dataset, X);
		int LU=0;
		for(int i=0;i<dataset.numInstances();i++){
			if(!dataset.instance(i).classIsMissing()){
				LU++;
			}
		}
		int U=Xi.length;
		int N=Xi[0].length;
		int M=Yj[0][0].length;
		double res_joinentropy = 0.0;
		for (int i=0;i<N;++i){
			for(int j=0; j<M;++j){	
				int XlandYlnum=0;int Ylnum=0;
				int Ynum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[1][k][j]){
						XlandYlnum++;
					}
					if(Yj[1][k][j]){
						Ylnum++;
					}
					if(Yj[0][k][j]){
						Ynum++;
					}
				}
				if(XlandYlnum!=0){	
					double ptemp=((double)(Ynum*XlandYlnum))/((double)(Ylnum*U));
					res_joinentropy=res_joinentropy-ptemp*(Math.log(ptemp) / Math.log((double)2));
				}
			}
		}
//		res_joinentropy=res_joinentropy/U;
		return res_joinentropy;
	}
	
	
	//求半监督条件熵
	public static double getSemiConditionalEntorpy(Instances dataset, boolean[] X, boolean[] Y){
		
		boolean[][][] Yj= getEquivalenceClassforSemi(dataset, Y);	
		boolean[][] Xi = getEquivalenceClassforlabel(dataset, X);
		int LU=0;
		for(int i=0;i<dataset.numInstances();i++){
			if(!dataset.instance(i).classIsMissing()){
				LU++;
			}
		}
		int U=Xi.length;
		int N=Xi[0].length;
		int M=Yj[0][0].length;
		double res_entropy=0.0;
		for (int j=0;j<M;++j){
			int Ylnum=0;int Ynum=0;
			for(int i=0; i<N;++i){	
				int XlandYlnum=0;
				Ynum=0;Ylnum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[1][k][j]){
						XlandYlnum++;
					}
					if(Yj[1][k][j]){
						Ylnum++;
					}
					if(Yj[0][k][j]){
						Ynum++;
					}
				}
				if(XlandYlnum!=0){	
					double conp=((double)XlandYlnum)/((double)Ylnum);
					res_entropy=res_entropy-Ynum*conp*(Math.log(conp)/ Math.log((double)2));
				}
			}
			if(Ylnum==0){
				Ynum=0;
				for(int k=0;k<U;k++){
					if(Yj[0][k][j])
						Ynum++;
				}
				double py=((double)Ynum)/((double)(U));
				res_entropy=res_entropy+Ynum*(Math.log(py)/ Math.log((double)2));
			}
		}
		res_entropy=res_entropy/U;
		
		
//		double Hxy;//Hxy2,//Hxy=getSemiJointEntropy(dataset,X,Y);	
		Instances ldataset=new Instances(dataset,0);
		for(int i=0;i<dataset.numInstances();i++){
			if(!dataset.instance(i).classIsMissing())
				ldataset.add(dataset.instance(i));
		}
//		if(ldataset.numInstances()>0)
//			Hxy=getJointEntropy(ldataset,X,Y);
//		else
//			Hxy=0;
		
//		double res_entropy2=Hxy-Hy;

		return res_entropy;
	}
	
	public static double getSemiRatioConditionalEntorpy(Instances dataset, boolean[] X, boolean[] Y){
		Instances ldataset=new Instances(dataset,0);
		for(int i=0;i<dataset.numInstances();i++){
			if(!dataset.instance(i).classIsMissing())
				ldataset.add(dataset.instance(i));
		}
		double Hx=getInformationEntorpy(ldataset,X);
		double Hy=getInformationEntorpy(dataset,Y);
		double res_entropy=getConditionalEntorpy(ldataset,X,Y);
		
		return (Hx-res_entropy)/Hy;
	}
	
	public static double getRatioConditionalEntorpy(Instances dataset, boolean[] X, boolean[] Y){
		double Hx=getInformationEntorpy(dataset,X);
		double Hy=getInformationEntorpy(dataset,Y);
		double res_entropy=getConditionalEntorpy(dataset,X,Y);
		
		return (Hx-res_entropy)/Hy;
	}
	
	//求条件熵conditional entropy
	//H(X|Y) boolean[]X,Y 包括决策属性
	public static double getConditionalEntorpy(Instances dataset, boolean[] X, boolean[] Y){
		boolean[][] Xi = getEquivalenceClass(dataset, X);	
		boolean[][] Yj = getEquivalenceClass(dataset, Y);
		int U = Xi.length;
		int N = Xi[0].length;
		int M = Yj[0].length;
		//计算条件熵
		double res_entropy = 0.0;
		for (int i=0;i<N;++i){			
			for(int j=0; j<M;++j){			
				int XandYnum=0;
				int Ynum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[k][j]){
						//System.out.println("##"+k+":"+i+":"+j);
						XandYnum++;}
					if(Yj[k][j])
						
						Ynum++;
				}
				//System.out.println(XandYnum+":"+Ynum);
				if(XandYnum !=0)
					res_entropy=res_entropy-XandYnum* (Math.log((double)XandYnum/Ynum) / Math.log((double)2));
					//注意加(double) 否则part1_sum/part2_sum = 0 熵为负无穷
			}
		}
		res_entropy=(double)res_entropy/U;
		return res_entropy;			
	}

	//求joint entropy
	//H(X,Y)
	public static double getJointEntropy(Instances dataset, boolean[] X, boolean[] Y){
		double Hy = getInformationEntorpy(dataset, Y);
		double Hxy = getConditionalEntorpy(dataset, X, Y);
		return Hy+Hxy;
	}
	
	//求CA(Class-Attribute) mutual information
	//CA(X:Y)
	public static double getCA(Instances dataset, boolean[] X, boolean[] Y){
		boolean[][] Xi = getEquivalenceClass(dataset, X);	
		boolean[][] Yj = getEquivalenceClass(dataset, Y);
		
		int U = Xi.length;
		int N = Xi[0].length;
		int M = Yj[0].length;
		//计算条件熵
		double res_CA = 0.0;
		for (int i=0;i<N;++i){				
			for(int j=0; j<M;++j){
				int XandYnum=0;
				int Ynum=0;
				int Xnum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[k][j])
						XandYnum++;
					if(Yj[k][j])
						Ynum++;
					if(Xi[k][i])
						Xnum++;
				}
				if(XandYnum !=0)
					res_CA=res_CA+XandYnum* (Math.log((XandYnum*U*1.0)/(Ynum*Xnum*1.0)) / Math.log((double)2));
					//注意加(double) 否则part1_sum/part2_sum = 0 熵为负无穷
			}
		}
		
		res_CA=(double)res_CA/U;
		return res_CA;	
	}
	public static double ClassDependent(Instances dataset, boolean[] X, boolean[] Y) {
		// TODO Auto-generated method stub
		boolean[][] Xi = getEquivalenceClass(dataset, X);	
		boolean[][] Yj = getEquivalenceClass(dataset, Y);

		
		int U = Xi.length;
		int N = Xi[0].length;
		int M = Yj[0].length;
		//计算互信息
		double res_entropy = 0.0;
		for (int i=0;i<N;++i){				
			for(int j=0; j<M;++j){
				int XandYnum=0;
				int Ynum=0;
				int Xnum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[k][j])
						XandYnum++;
					if(Xi[k][i])
						Xnum++;
					if(Yj[k][j])
						Ynum++;
				}
				if(XandYnum !=0)
					res_entropy=res_entropy+XandYnum* (Math.log((double)(XandYnum*U)/(double)(Ynum*Xnum)) / Math.log((double)2));
					//注意加(double) 否则part1_sum/part2_sum = 0 熵为负无穷 不加负号
			}
		}
		
		res_entropy=(double)res_entropy/U;
		//System.out.println(res_entropy);
		return res_entropy;	

	}
	//求CAIR Class-Attribute Interdependence Redundancy
	//CAIR(X,Y)=CA(X,Y)/H(X,Y)
	public static double getCAIR(Instances dataset, boolean[] X, boolean[] Y){
		double Hx_y = getJointEntropy(dataset,X,Y);
		double CA = getCA(dataset,X,Y);
		return CA/Hx_y;
	}
	
	//求information gain(mutual information)
	//IG(X|Y)
	public static double getIG(Instances dataset, boolean[] X, boolean[] Y){
		double Hx = getInformationEntorpy(dataset, X);
		double Hxy = getConditionalEntorpy(dataset, X, Y);
		return Hx-Hxy;
	}
	//求symmetrical uncertainty
	//SU(X,Y)
	public static double getSU(Instances dataset, boolean[] X, boolean[] Y){	
		boolean[][] Xi = getEquivalenceClass(dataset, X);
		boolean[][] Yj = getEquivalenceClass(dataset, Y);
		int U = Xi.length;
		int N = Xi[0].length;
		double Hx = 0.0;
		for(int i=0;i<N;++i){
			int sum=0;
			for(int k=0;k<U;++k){
				if(Xi[k][i])
				sum++;
			}
			if(sum!=(double)0){
				Hx -= sum* (Math.log((double)sum/U) / Math.log((double)2));}
		}
		Hx=(double)Hx/(double)U;
		U = Yj.length;
		N = Yj[0].length;
		double Hy = 0.0;
		for(int i=0;i<N;++i){
			int sum=0;
			for(int k=0;k<U;++k){
				if(Yj[k][i])
				sum++;
			}
			if(sum!=(double)0){
				Hy -= sum* (Math.log((double)sum/U) / Math.log((double)2));}
		}
		Hy=(double)Hy/(double)U;
		U = Xi.length;
		N = Xi[0].length;
		int M = Yj[0].length;
		//计算条件熵
		double res_entropy = 0.0;
		for (int i=0;i<N;++i){				
			for(int j=0; j<M;++j){
				int XandYnum=0;
				int Ynum=0;
				for(int k=0;k<U;++k){
					if(Xi[k][i]&Yj[k][j])
						XandYnum++;
					if(Yj[k][j])
						Ynum++;
				}
				if(XandYnum !=0)
					res_entropy=res_entropy-XandYnum* (Math.log((double)XandYnum/Ynum) / Math.log((double)2));
					//注意加(double) 否则part1_sum/part2_sum = 0 熵为负无穷
			}
		}
		
		res_entropy=(double)res_entropy/U;

		double Hxy = res_entropy;
		return 2.0*(Hx-Hxy)/(Hx+Hy);
	}
	public static void main(String[] args) throws Exception, IOException {
		// TODO Auto-generated method stub
		//String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/fuzzy/fuzzy-ex2.arff";
		String fn = "C:/Users/Eric/Desktop/2011秋冬/Code/Xreducer/data/Data/sonar.arff";
		Instances data = new Instances(new FileReader(fn));
		data.setClassIndex(data.numAttributes()-1); //设置决策属性索引
		
		
		
		weka.filters.supervised.attribute.Discretize sd = new weka.filters.supervised.attribute.Discretize();
		try {
			sd.setInputFormat(data);
			data = Filter.useFilter(data , sd);
		} catch (Exception e) {
				// TODO Auto-generated catch block
			e.printStackTrace();}
			
			
		int att = data.numAttributes(); //带决策属性
		boolean[] D = Utils.Instances2DecBoolean(data);
		boolean[] A = Utils.Instances2FullBoolean(data);
		boolean[] B = new boolean[att];
		//B[1] = true; //a
		//B[2] = true; //a
		B[1] = true; //a
		//B[3] = true; //a
		//B[21] = true; //b
		//B[26] = true; //b
		//B[33] = true; //b
		//B[2] = true; //c
		//B[3] = true; //d
		//boolean[] p = getPOS_pX(data, D, B);
		//System.out.println(Arrays.toString(p));
		//boolean[] b = getBND_pX(data, D, B);
		//System.out.println(Arrays.toString(b));
		double a1 = getGranulation(data,D,B);
		//double b1 = getCA(data,D,B);
		System.out.println(a1);
		
		
	}
	public static void show(boolean[][] x){
		int n = x.length;
		int m = x[0].length;
		for(int i=0;i<n;++i){
			String p = "";
			for(int j=0;j<m;++j){
				if(x[i][j])
					p = p+"1";
				else
					p = p+"0";
			}
			System.out.println(i+":"+p);
		}
		
	}


		
}
