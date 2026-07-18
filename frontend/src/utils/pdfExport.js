import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export async function exportToPDF(elementRef, filename = 'report.pdf') {
  if (!elementRef || !elementRef.current) {
    console.error("No element provided for PDF export");
    return;
  }

  try {
    // Save original styles that might interfere with PDF layout
    const originalStyles = {
      overflow: elementRef.current.style.overflow,
      height: elementRef.current.style.height,
    };

    // Prepare element for full capture
    elementRef.current.style.overflow = 'visible';
    elementRef.current.style.height = 'auto';

    // HTML2Canvas capture
    const canvas = await html2canvas(elementRef.current, {
      scale: 2, // Higher resolution
      useCORS: true,
      logging: false,
      backgroundColor: '#111827', // Match the main app background
      windowWidth: elementRef.current.scrollWidth,
      windowHeight: elementRef.current.scrollHeight
    });

    // Restore styles
    elementRef.current.style.overflow = originalStyles.overflow;
    elementRef.current.style.height = originalStyles.height;

    const imgData = canvas.toDataURL('image/png');
    const pdf = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
    
    let heightLeft = pdfHeight;
    let position = 0;

    // Add first page
    pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
    heightLeft -= pdf.internal.pageSize.getHeight();

    // Handle multipage documents
    while (heightLeft > 0) {
      position = heightLeft - pdfHeight;
      pdf.addPage();
      pdf.addImage(imgData, 'PNG', 0, position, pdfWidth, pdfHeight);
      heightLeft -= pdf.internal.pageSize.getHeight();
    }

    pdf.save(filename);
  } catch (error) {
    console.error('Error exporting PDF:', error);
    throw new Error('Failed to generate PDF report.');
  }
}
