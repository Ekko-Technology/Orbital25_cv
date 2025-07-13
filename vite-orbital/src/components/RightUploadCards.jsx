import React from 'react';
import {Button, Spinner, Card} from 'react-bootstrap';
import {Edit} from 'lucide-react'

function RightUploadCard({ onUpload, loading, selectedFile, modifiedImageUrl, modifiedImageRef, canvasRef, handleImageClick, gameStarted, foundDifferences, differences, clickAttempts, MAX_WRONG_CLICKS}) {
    return (
        <Card className="h-100 shadow-sm rounded-4">
            <Card.Header className="text-center bg-primary text-white">
                Modified Image (Click on the image below!)
            </Card.Header>

            <Card.Body className="cardBodyStyle" style={{ position: 'relative' }}>
                {modifiedImageUrl ? (
                    <>
                        <img
                            ref={modifiedImageRef}
                            key={modifiedImageUrl}
                            src={modifiedImageUrl}
                            alt="Modified"
                            className="img-fluid"
                            style={{ cursor: gameStarted && foundDifferences.size < differences.length && clickAttempts.filter(a => a.type === 'wrong').length < MAX_WRONG_CLICKS ? 'pointer' : 'default' }}
                            onClick={handleImageClick}
                        />
                        <canvas
                            ref={canvasRef}
                            style={{
                            position: 'absolute', 
                            width: '100%',
                            height: '100%', 
                            pointerEvents: 'none',
                            }}
                        />
                    </>
                ) : (
                    <div className="text-center text-muted d-flex flex-column align-items-center">
                    <Edit size={64} className="mb-3" />
                    <p className="mb-0">Modified image will appear here</p>
                    <Button
                        variant="success"
                        onClick={onUpload}
                        disabled={!selectedFile || loading} 
                        className="mt-3"
                    >
                        {loading ? <Spinner animation="border" size="sm" /> : 'Generate Modified Image'}
                    </Button>
                </div>
            )}
            </Card.Body>
        </Card>
    );
}


export default RightUploadCard;