package cluster.ABC;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

import weka.core.DistanceFunction;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NormalizableDistance;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

public class KernelDistances extends NormalizableDistance implements
		DistancewithPar,Cloneable, TechnicalInformationHandler {

	double par=0.1;

	/**
	 * 
	 */
	private static final long serialVersionUID = 5913677295488838544L;

	public KernelDistances() {
		    super();
    }

	public KernelDistances(Instances data) {
	    super(data);
	}

	public void setpar(double p){
		par=p;
	}
	
	
	@Override
	public String getRevision() {
		// TODO Auto-generated method stub
		 return RevisionUtils.extract("$Revision: 0.01 $");
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
	    TechnicalInformation 	result;
	    
	    result = new TechnicalInformation(Type.MISC);
	    result.setValue(Field.AUTHOR, "Wang Wentao");
	    result.setValue(Field.TITLE, "Dissimilarity measure for K-modes");

	    return result;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return 
        " Dissimilarity measure for K-modes\n\n"
      + getTechnicalInformation().toString();
	}

	@Override
	protected double updateDistance(double currDist, double diff) {
		// TODO Auto-generated method stub
		return 0;
	}

	public double getunitdis(){
		//return 2*Math.exp(0)-2*Math.exp(-1/(2*par*par));
		return Math.pow(1+1,par);
	}
	
	public double getunitdis(double unit){
	//	return Math.exp(-unit/(2*par*par));
		return Math.pow(unit+1,par);
	}
	
	
	public double getsimilarity(Instance first, Instance second){
	    double result=0;
	    double sum1first=0;double sumsecond=0;
	    int N = first.numAttributes();
	    for(int i=0;i<N;++i){
			result+=Math.pow(first.value(i)-second.value(i),2);
	    }
		return Math.exp(-result/(2*par*par));
	  }
	
	  /**
	   * Calculates the distance between two instances.
	   * 
	   * @param first 	the first instance
	   * @param second 	the second instance
	   * @return 		the distance between the two given instances
	   */
	  public double distance(Instance first, Instance second) {
	    double result=0;
	    double sum1first=0;double sumsecond=0;
	    int N = first.numAttributes();
		return 2*Math.exp(0)-2*getsimilarity(first,second);
	    //return  getsimilarity(first, first)+getsimilarity(second, second)--2*getsimilarity(first,second);
	  }
	
	/**
	 * @param args
	 * @throws IOException 
	 * @throws FileNotFoundException 
	 */
	public static void main(String[] args) throws FileNotFoundException, IOException {
		// TODO Auto-generated method stub
		String path = "C:/Users/Eric/Desktop/2011Çï¶¬/Code/Xreducer/data/Data/vote.arff";
		Instances m_data = new Instances(new FileReader(path));
		System.out.println(m_data.instance(0));
		System.out.println(m_data.instance(1));
		DistanceFunction dk = new KernelDistances();
		
		System.out.println(dk.distance(m_data.instance(2), m_data.instance(1)));
	}

}
