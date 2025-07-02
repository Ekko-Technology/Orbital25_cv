import React from 'react';
import {Button, Form, Spinner, Card} from 'react-bootstrap';
import {Image as ImageIcon} from 'lucide-react'

function LeftUploadCard({ HeaderText, onFileSelect, onUpload, loading, selectedFile, fileInputRef, triggerFileInput, originalImageUrl, modifiedImageUrl}) {
    return (
        <Card className="h-100 shadow-sm">
            <Card.Header className="text-center bg-dark text-white">
                {HeaderText}
            </Card.Header>
            
            <Card.Body className="cardBodyStyle">

                {/* Hidden file input */}
                <Form>
                    <Form.Control
                        type="file"
                        accept="image/*"
                        onChange={onFileSelect}
                        ref={fileInputRef}
                        className="d-none" 
                    />
                </Form>

                {/* Show uploaded image if available */}
                {originalImageUrl ? (
                    <div className="text-center mb-3">
                        <img
                        src={originalImageUrl}
                        alt="Original Upload"
                        className="img-fluid"
                        style={{ objectFit: 'contain' }}
                        />
                    </div>
                ) : (               
                    // Otherwise show upload icon and prompt
                    <div className="text-center text-muted d-flex flex-column align-items-center">
                        <ImageIcon size={64} className='mb-3' />
                        <Button
                            variant="link"
                            className="p-0 border-0 text-decoration-none" onClick={triggerFileInput} 
                        >
                            <p className="mb-0">
                                Upload an image to begin
                            </p>
                        </Button>
                    </div>
                )}

                {/* upload Button (only if file selected) */}
                {selectedFile && !modifiedImageUrl && (
                    <div className="mt-3">
                        {loading ? (
                        <Spinner animation="border" size="sm" />) : (
                            <>
                                <div>✔ Upload Completed</div>
                            </>
                        )}
                    </div>
                )}
            </Card.Body>
        </Card>
    );
}


export default LeftUploadCard;