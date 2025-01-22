using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;

public class GameObject
{
    public Vector2 Position { get; set; }
    public Vector2 Velocity { get; set; }
    public Vector2 Size { get; set; }
    public bool IsGrounded { get; private set; }

    public float gravity { private get; set; }
    private float terminalVelocity;

    private Texture2D texture;

    public GameObject(Vector2 position, Vector2 size, float gravity = 9.8f, float terminalVelocity = 1000f)
    {
        Position = position;
        Velocity = Vector2.Zero;
        Size = size;
        this.gravity = gravity;
        this.terminalVelocity = terminalVelocity;
    }

    public void LoadTexture(GraphicsDevice graphicsDevice, Color color)
    {
        // Create a simple colored texture for the object
        texture = new Texture2D(graphicsDevice, (int)Size.X, (int)Size.Y);
        Color[] data = new Color[(int)Size.X * (int)Size.Y];
        for (int i = 0; i < data.Length; ++i) data[i] = color;
        texture.SetData(data);
    }

    public void Update(GameTime gameTime, float groundLevel)
    {
        float deltaTime = (float)gameTime.ElapsedGameTime.TotalSeconds;

        // Apply gravity if not grounded
        if (!IsGrounded)
        {
            Velocity += new Vector2(0, gravity);

            // Clamp velocity to terminal velocity
            if (Velocity.Y > terminalVelocity)
                Velocity = new Vector2(Velocity.X, terminalVelocity);
        }

        // Update position
        Position += Velocity * deltaTime;

        // Check if we've hit the ground
        if (Position.Y + Size.Y >= groundLevel)
        {
            Position = new Vector2(Position.X, groundLevel - Size.Y);
            Velocity = new Vector2(Velocity.X, 0); // Stop vertical movement
            IsGrounded = true;
        }
        else
        {
            IsGrounded = false;
        }
    }

    public void Draw(SpriteBatch spriteBatch)
    {
        if (texture != null)
        {
            spriteBatch.Draw(texture, Position, Color.White);
        }
    }

    public void ApplyForce(Vector2 force)
    {
        Velocity += force;
    }
    public void ApplyImpulse(Vector2 impulse)
    {
        Velocity = impulse;
    }
    public void Move(Vector2 move)
    {
        Position += move;
    }
    public bool Intersects(GameObject other)
    {
        Rectangle thisRect = new Rectangle((int)Position.X, (int)Position.Y, (int)Size.X, (int)Size.Y);
        Rectangle otherRect = new Rectangle((int)other.Position.X, (int)other.Position.Y, (int)other.Size.X, (int)other.Size.Y);
        return thisRect.Intersects(otherRect);
    }

}
