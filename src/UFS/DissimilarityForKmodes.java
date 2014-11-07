package UFS;

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

public class DissimilarityForKmodes extends NormalizableDistance implements
		Cloneable, TechnicalInformationHandler {



	/**
	 * 
	 */
	private static final long serialVersionUID = 5913677295488838544L;

	public DissimilarityForKmodes() {
		    super();
    }

	public DissimilarityForKmodes(Instances data) {
	    super(data);
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

	
	
	  /**
	   * Calculates the distance between two instances.
	   * 
	   * @param first 	the first instance
	   * @param second 	the second instance
	   * @return 		the distance between the two given instances
	   */
	  public double distance(Instance first, Instance second) {
	    int result=0;
	    int N = first.numAttributes();
	    for(int i=0;i<N;++i){
	    	if(Double.isNaN(first.value(i))||Double.isNaN(second.value(i))||first.value(i)!=second.value(i))
	    		result++;
	    }
	    
	    return (double) result;
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
		DistanceFunction dk = new DissimilarityForKmodes();
		
		System.out.println(dk.distance(m_data.instance(2), m_data.instance(1)));
	}

}
